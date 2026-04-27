from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from corpuscraft.config import PipelineType


@dataclass
class ContentProfile:
    # Tier 0 — always populated from PreprocessedPDF (no detection needed)
    extension: str          # lowercase with dot: ".pdf", ".docx", etc.
    page_count: int
    is_scanned: bool
    encrypted: bool
    page_size: str          # raw pdfinfo string, e.g. "595 x 842 pts"

    # Tier 1 — "basic" detection via pymupdf/fitz (None = not measured)
    image_ratio: float | None = None          # image block area / page area, 0.0–1.0
    table_count: int | None = None            # tables found via page.find_tables()
    has_formula_heuristic: bool | None = None # LaTeX regex + unicode math chars
    is_multi_column: bool | None = None       # bimodal X-position histogram test
    text_density: float | None = None         # avg chars/page across sampled pages

    # Tier 2 — "enhanced" detection via YOLO (None = YOLO was not run)
    yolo_label_counts: dict[str, int] | None = None  # e.g. {"Table": 3, "Formula": 1}
    yolo_avg_confidence: float | None = None

    detection_level: Literal["none", "basic", "enhanced"] = "none"


@dataclass
class RoutingResult:
    pipeline: PipelineType
    reason: str             # human-readable, safe to log
    confidence: float       # 0.0 (pure guess) – 1.0 (certain)
    alternatives: list[PipelineType] = field(default_factory=list)
    profile: ContentProfile | None = None
