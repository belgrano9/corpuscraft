from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from corpuscraft.preprocessing.models import PreprocessedPDF
from corpuscraft.routing.models import ContentProfile

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Formula detection constants
# ---------------------------------------------------------------------------
_FORMULA_PATTERNS: list[str] = [
    r"\\(?:frac|sum|int|sqrt|lim|prod|alpha|beta|gamma|delta|epsilon|theta"
    r"|lambda|mu|sigma|omega|pi|phi|psi|nabla|partial|infty)\b",
    r"\$[^$\n]{1,80}\$",    # inline LaTeX $...$
    r"\\\[.*?\\\]",          # display LaTeX \[...\]
    r"\\\(.*?\\\)",          # inline LaTeX \(...\)
]
_FORMULA_RE = re.compile("|".join(_FORMULA_PATTERNS), re.DOTALL)

# Unicode Mathematical Operators block U+2200–U+22FF, plus common math symbols
_MATH_CHARS: frozenset[str] = frozenset(
    chr(cp) for cp in range(0x2200, 0x2300)
) | frozenset("²³±×÷√∞≈≠≤≥∑∏∫∂∇")

# YOLO DocLayNet label index → name
_YOLO_LABELS: dict[int, str] = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture",
    7: "Section-header", 8: "Table", 9: "Text", 10: "Title",
}

_MULTI_COLUMN_X_BINS = 10
_MULTI_COLUMN_PEAK_THRESHOLD = 0.30   # peak must be >30 % of max bin to count
_MIN_TEXT_BLOCKS_FOR_LAYOUT = 4       # skip layout analysis on sparse pages


class ContentDetector:
    """
    Inspects a document and returns a ContentProfile for use by RoutingRules.

    Detection levels:
      "none"     — use only what PreprocessedPDF already provides; no file I/O
      "basic"    — open PDF with pymupdf/fitz (core dep); detect tables, images,
                   multi-column layout, text density, formula heuristics
      "enhanced" — run basic first, then run YOLO on rasterized pages for
                   label-level detection (requires [yolo] extra)
    """

    def __init__(
        self,
        level: Literal["none", "basic", "enhanced"] = "basic",
        sample_pages: int = 5,
        yolo_model_id: str = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
        yolo_confidence: float = 0.2,
    ) -> None:
        self.level = level
        self.sample_pages = sample_pages
        self.yolo_model_id = yolo_model_id
        self.yolo_confidence = yolo_confidence
        self._yolo_model = None

        if level == "enhanced":
            try:
                from doclayout_yolo import YOLOv10  # type: ignore[import-untyped]
                self._yolo_model = YOLOv10(yolo_model_id)
            except ImportError as exc:
                raise ImportError(
                    "detection_level='enhanced' requires the [yolo] extra. "
                    "Install with: uv pip install corpuscraft[yolo]"
                ) from exc

    def detect(self, preprocessed: PreprocessedPDF) -> ContentProfile:
        base = {
            "extension": preprocessed.source_path.suffix.lower(),
            "page_count": preprocessed.metadata.page_count,
            "is_scanned": preprocessed.is_scanned,
            "encrypted": preprocessed.metadata.encrypted,
            "page_size": preprocessed.metadata.page_size,
            "detection_level": self.level,
        }

        ext = base["extension"]

        # Non-PDF, encrypted, or scanned: skip content detection
        # (fitz would fail on encrypted; scanned pages yield no text blocks)
        if ext != ".pdf" or base["encrypted"] or base["is_scanned"] or self.level == "none":
            return ContentProfile(**base)

        path = preprocessed.parser_input()
        return self._detect_with_fitz(path, base)

    # ------------------------------------------------------------------
    # fitz-based detection
    # ------------------------------------------------------------------

    def _detect_with_fitz(self, path: Path, base: dict) -> ContentProfile:
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(str(path))
        pages = list(range(min(self.sample_pages, len(doc))))

        if not pages:
            doc.close()
            return ContentProfile(**base)

        image_ratio = _compute_image_ratio(doc, pages)
        table_count = _count_tables(doc, pages)
        text_density = _compute_text_density(doc, pages)
        has_formula = _detect_formulas(doc, pages)
        is_multi_column = _detect_multi_column(doc, pages)
        doc.close()

        profile = ContentProfile(
            **base,
            image_ratio=image_ratio,
            table_count=table_count,
            text_density=text_density,
            has_formula_heuristic=has_formula,
            is_multi_column=is_multi_column,
            detection_level="basic",
        )

        if self.level == "enhanced" and self._yolo_model is not None:
            # Build a fake PreprocessedPDF-like object just to pass path info
            label_counts, avg_conf = _run_yolo(
                path, self._yolo_model, self.yolo_confidence, pages
            )
            profile.yolo_label_counts = label_counts
            profile.yolo_avg_confidence = avg_conf
            profile.detection_level = "enhanced"

        return profile


# ---------------------------------------------------------------------------
# Private detection helpers (module-level, not methods)
# ---------------------------------------------------------------------------

def _compute_image_ratio(doc: object, pages: list[int]) -> float:
    ratios: list[float] = []
    for idx in pages:
        page = doc[idx]  # type: ignore[index]
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            continue
        blocks = page.get_text("dict")["blocks"]
        img_area = sum(
            (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
            for b in blocks
            if b.get("type") == 1  # type 1 = image block in fitz
        )
        ratios.append(img_area / page_area)
    return statistics.mean(ratios) if ratios else 0.0


def _count_tables(doc: object, pages: list[int]) -> int:
    total = 0
    for idx in pages:
        page = doc[idx]  # type: ignore[index]
        try:
            finder = page.find_tables()
            total += len(finder.tables)
        except AttributeError:
            # pymupdf < 1.23 does not have find_tables(); degrade gracefully
            break
    return total


def _compute_text_density(doc: object, pages: list[int]) -> float:
    densities: list[float] = []
    for idx in pages:
        text = doc[idx].get_text("text")  # type: ignore[index]
        densities.append(float(len(text)))
    return statistics.mean(densities) if densities else 0.0


def _detect_formulas(doc: object, pages: list[int]) -> bool:
    # Heuristic — misses bitmap-rendered formulas; YOLO enhanced level is authoritative
    for idx in pages:
        text = doc[idx].get_text("text")  # type: ignore[index]
        if _FORMULA_RE.search(text):
            return True
        if any(c in _MATH_CHARS for c in text):
            return True
    return False


def _detect_multi_column(doc: object, pages: list[int]) -> bool:
    multi_column_votes = 0
    for idx in pages:
        page = doc[idx]  # type: ignore[index]
        page_width = page.rect.width
        if page_width == 0:
            continue

        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b.get("type") == 0 and b.get("lines")]
        if len(text_blocks) < _MIN_TEXT_BLOCKS_FOR_LAYOUT:
            continue

        # Normalise X-centre of each text block to [0, 1]
        x_centres = [
            ((b["bbox"][0] + b["bbox"][2]) / 2) / page_width
            for b in text_blocks
        ]

        hist = [0] * _MULTI_COLUMN_X_BINS
        for x in x_centres:
            bin_idx = min(int(x * _MULTI_COLUMN_X_BINS), _MULTI_COLUMN_X_BINS - 1)
            hist[bin_idx] += 1

        max_count = max(hist)
        if max_count == 0:
            continue

        threshold = _MULTI_COLUMN_PEAK_THRESHOLD * max_count
        peak_regions = 0
        in_peak = False
        for count in hist:
            if count > threshold and not in_peak:
                in_peak = True
                peak_regions += 1
            elif count <= threshold:
                in_peak = False

        if peak_regions >= 2:
            multi_column_votes += 1

    return multi_column_votes > len(pages) / 2


def _run_yolo(
    path: Path,
    model: object,
    confidence: float,
    pages: list[int],
) -> tuple[dict[str, int], float]:
    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    SCALE = 150 / 72.0
    label_counts: dict[str, int] = {}
    confidences: list[float] = []

    pdf_doc = pdfium.PdfDocument(str(path))
    for idx in pages:
        if idx >= len(pdf_doc):
            break
        pil = pdf_doc[idx].render(scale=SCALE).to_pil()
        results = model.predict(  # type: ignore[union-attr]
            pil, imgsz=1024, conf=confidence, device="cpu", verbose=False
        )
        boxes = results[0].boxes if results else None
        if boxes is None:
            continue
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            label = _YOLO_LABELS.get(int(cls_id), "Unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
            confidences.append(float(conf))

    avg_conf = statistics.mean(confidences) if confidences else 0.0
    return label_counts, avg_conf
