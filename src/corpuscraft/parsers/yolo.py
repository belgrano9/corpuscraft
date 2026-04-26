"""
DocLayout-YOLO based PDF parser for CorpusCraft.

Uses DocLayout-YOLO to detect layout regions (titles, section headers, tables,
figures, etc.) in PDF pages, then extracts text from each region via pdfplumber
in reading order.

Limitations:
- Native PDFs only — scanned PDFs yield empty text; use DoclingParser with OCR instead.
- Reading order is naive top-y → left-x; 2-column layouts will interleave columns.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from loguru import logger

from corpuscraft.models import ParsedDocument

RENDER_DPI = 150
_SCALE = RENDER_DPI / 72.0  # pixels per PDF point

_SKIP_LABELS = {"Page-header", "Page-footer"}
_HEADING_LABELS = {"Title": "#", "Section-header": "##"}
_BLOCK_LABELS = {"Table": "[TABLE]", "Picture": "[FIGURE]"}

# DocStructBench class index → label name
DOCSTRUCTBENCH_NAMES: dict[int, str] = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


class YoloParser:
    """PDF parser using DocLayout-YOLO for layout detection + pdfplumber for text extraction."""

    SUPPORTED_FORMATS = {".pdf"}

    def __init__(
        self,
        yolo_model: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        yolo_confidence: float = 0.2,
        device: str = "cpu",
    ):
        """
        Initialize the YOLO-based PDF parser.

        Args:
            yolo_model: HuggingFace model ID or local path for DocLayout-YOLO weights.
            yolo_confidence: Minimum detection confidence threshold (0–1).
            device: Inference device — "cpu", "cuda", or "mps".
        """
        from doclayout_yolo import YOLOv10

        logger.info(f"Loading DocLayout-YOLO model: {yolo_model}")
        self._model = YOLOv10(yolo_model)
        self._conf = yolo_confidence
        self._device = device

    def parse_file(self, file_path: str | Path) -> ParsedDocument | None:
        """
        Parse a single PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with structured markdown content and metadata,
            or None if the file format is unsupported or parsing fails.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"YoloParser only supports PDF files, got: {file_path.suffix}")
            return None

        try:
            return self._parse_pdf(file_path)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        pdf_pdfium = pdfium.PdfDocument(str(path))
        num_pages = len(pdf_pdfium)
        page_texts: list[str] = []

        with pdfplumber.open(path) as pdf_plumber:
            for page_idx in range(num_pages):
                plumber_page = pdf_plumber.pages[page_idx]
                pdfium_page = pdf_pdfium[page_idx]

                # Render to PIL for YOLO inference
                bitmap = pdfium_page.render(scale=_SCALE)
                pil_image = bitmap.to_pil()

                results = self._model.predict(
                    pil_image,
                    imgsz=1024,
                    conf=self._conf,
                    device=self._device,
                    verbose=False,
                )

                boxes = results[0].boxes if results else None
                if boxes is None or len(boxes) == 0:
                    text = plumber_page.extract_text() or ""
                    if text.strip():
                        page_texts.append(text.strip())
                    else:
                        logger.warning(
                            f"Page {page_idx + 1}: no YOLO detections and no extractable text — "
                            "this may be a scanned PDF; use DoclingParser with OCR instead."
                        )
                    continue

                # Sort detections top-to-bottom, then left-to-right
                detections = sorted(
                    zip(boxes.cls.tolist(), boxes.xyxy.tolist()),
                    key=lambda d: (d[1][1], d[1][0]),  # y1, x1
                )

                segments: list[str] = []
                for cls_id, (x1, y1, x2, y2) in detections:
                    label = DOCSTRUCTBENCH_NAMES.get(int(cls_id), "Text")
                    if label in _SKIP_LABELS:
                        continue

                    # Convert pixel coordinates → PDF points
                    bbox = (x1 / _SCALE, y1 / _SCALE, x2 / _SCALE, y2 / _SCALE)
                    text = plumber_page.crop(bbox).extract_text() or ""
                    if not text.strip():
                        continue

                    if label in _HEADING_LABELS:
                        prefix = _HEADING_LABELS[label]
                        segments.append(f"{prefix} {text.strip()}")
                    elif label in _BLOCK_LABELS:
                        tag = _BLOCK_LABELS[label]
                        segments.append(f"{tag}\n{text.strip()}")
                    else:
                        segments.append(text.strip())

                if segments:
                    page_texts.append("\n\n".join(segments))

        content = "\n\n---\n\n".join(page_texts)

        metadata = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "format": "pdf",
            "file_size": path.stat().st_size,
            "num_pages": num_pages,
            "pipeline": "yolo",
        }

        logger.info(f"Successfully parsed {path.name}: {len(content)} characters")
        return ParsedDocument(content=content, metadata=metadata)

    def parse_folder(
        self,
        folder_path: str | Path,
        recursive: bool = True,
    ) -> list[ParsedDocument]:
        """
        Parse all PDF files in a folder.

        Args:
            folder_path: Path to the directory containing PDF files.
            recursive: If True, search subdirectories recursively.

        Returns:
            List of ParsedDocument objects (failed parses are excluded).

        Raises:
            NotADirectoryError: If the path is not a directory.
        """
        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        glob = folder_path.rglob if recursive else folder_path.glob
        files = list(glob("*.pdf"))
        logger.info(f"Found {len(files)} PDF files in {folder_path}")

        docs = [d for f in files if (d := self.parse_file(f)) is not None]
        logger.info(f"Successfully parsed {len(docs)}/{len(files)} documents")
        return docs
