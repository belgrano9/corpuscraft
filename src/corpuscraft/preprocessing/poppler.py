from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from corpuscraft.preprocessing.base import BasePreprocessor
from corpuscraft.preprocessing.models import PDFMetadata, PreprocessedPDF


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"poppler binary '{name}' not found on PATH. "
            "Install poppler-utils and ensure its bin/ directory is in PATH. "
            "Windows builds: https://github.com/oschwartz10612/poppler-windows/releases"
        )
    return path


class PopplerPreprocessor(BasePreprocessor):
    """
    Preprocesses PDF files using poppler utilities before parsing.

    Operations (all optional except inspect + probe, which always run):
      - inspect:   pdfinfo  → extracts metadata (page count, encryption, dimensions)
      - probe:     pdftotext → detects whether a native text layer exists
      - clean:     pdftocairo -pdf → re-renders PDF, stripping invisible text,
                   annotations, comments, form fields and embedded JavaScript
      - split:     pdfseparate → one PDF per page, enables per-page parallelism
      - rasterize: pdftoppm → renders pages to PNG/JPEG at a given DPI

    Example usage::

        from pathlib import Path
        from corpuscraft.preprocessing.poppler import PopplerPreprocessor
        from corpuscraft.config import PreprocessingConfig

        pre = PopplerPreprocessor(PreprocessingConfig(clean=True, rasterize=True, raster_dpi=150))
        result = pre.run(Path("paper.pdf"), Path("outputs/preprocessed/paper"))

        print(result)                    # PreprocessedPDF(...)
        print(result.is_scanned)         # True / False
        print(result.parser_input())     # cleaned PDF path (or original if clean=False)
        print(result.page_images)        # [Path("outputs/.../paper-1.png"), ...]
    """

    def __init__(
        self,
        clean: bool = True,
        split: bool = False,
        rasterize: bool = False,
        raster_dpi: int = 150,
        raster_format: str = "png",
        scanned_text_threshold: int = 100,
    ) -> None:
        self.clean = clean
        self.split = split
        self.rasterize = rasterize
        self.raster_dpi = raster_dpi
        self.raster_format = raster_format
        self.scanned_text_threshold = scanned_text_threshold

    def run(self, pdf_path: Path, output_dir: Path) -> PreprocessedPDF:
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._inspect(pdf_path)
        is_scanned = self._probe_text(pdf_path)

        cleaned_pdf = self._clean(pdf_path, output_dir) if self.clean else None
        page_pdfs = self._split(pdf_path, output_dir) if self.split else []
        page_images = self._rasterize(pdf_path, output_dir) if self.rasterize else []

        return PreprocessedPDF(
            source_path=pdf_path,
            output_dir=output_dir,
            metadata=metadata,
            is_scanned=is_scanned,
            cleaned_pdf=cleaned_pdf,
            page_pdfs=page_pdfs,
            page_images=page_images,
        )

    # ------------------------------------------------------------------
    # Private operations
    # ------------------------------------------------------------------

    def _inspect(self, pdf_path: Path) -> PDFMetadata:
        _require_binary("pdfinfo")
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        raw: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                raw[key.strip().lower().replace(" ", "_")] = value.strip()

        return PDFMetadata(
            page_count=int(raw.get("pages", 0)),
            encrypted=raw.get("encrypted", "no").lower() == "yes",
            pdf_version=raw.get("pdf_version", ""),
            title=raw.get("title", ""),
            author=raw.get("author", ""),
            page_size=raw.get("page_size", ""),
            raw=raw,
        )

    def _probe_text(self, pdf_path: Path) -> bool:
        """Returns True when the PDF has no meaningful native text (i.e. is scanned)."""
        _require_binary("pdftotext")
        result = subprocess.run(
            ["pdftotext", "-q", str(pdf_path), "-"],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip()) < self.scanned_text_threshold

    def _clean(self, pdf_path: Path, output_dir: Path) -> Path:
        """Re-render via pdftocairo to strip invisible text, annotations and comments."""
        _require_binary("pdftocairo")
        out = output_dir / f"{pdf_path.stem}_clean.pdf"
        subprocess.run(
            ["pdftocairo", "-pdf", str(pdf_path), str(out)],
            capture_output=True,
            check=True,
        )
        return out

    def _split(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """Split PDF into one file per page using pdfseparate."""
        _require_binary("pdfseparate")
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)
        prefix = str(pages_dir / f"{pdf_path.stem}_%d.pdf")
        subprocess.run(
            ["pdfseparate", str(pdf_path), prefix],
            capture_output=True,
            check=True,
        )
        return sorted(pages_dir.glob(f"{pdf_path.stem}_*.pdf"))

    def _rasterize(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """Render pages to images using pdftoppm."""
        _require_binary("pdftoppm")
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        prefix = str(images_dir / pdf_path.stem)
        subprocess.run(
            [
                "pdftoppm",
                f"-{self.raster_format}",
                "-r", str(self.raster_dpi),
                str(pdf_path),
                prefix,
            ],
            capture_output=True,
            check=True,
        )
        return sorted(images_dir.glob(f"{pdf_path.stem}-*.{self.raster_format}"))
