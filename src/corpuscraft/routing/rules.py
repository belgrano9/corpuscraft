from __future__ import annotations

from corpuscraft.config import PipelineType
from corpuscraft.routing.models import ContentProfile, RoutingResult

# ---------------------------------------------------------------------------
# Non-PDF extension routing table: extension → (primary, alternatives)
# ---------------------------------------------------------------------------
_NON_PDF_ROUTES: dict[str, tuple[PipelineType, list[PipelineType]]] = {
    ".docx": (PipelineType.python_docx, [PipelineType.mammoth, PipelineType.markitdown, PipelineType.standard]),
    ".pptx": (PipelineType.python_pptx, [PipelineType.markitdown, PipelineType.standard]),
    ".html": (PipelineType.standard, [PipelineType.markitdown]),
    ".htm":  (PipelineType.standard, [PipelineType.markitdown]),
    ".md":   (PipelineType.standard, []),
    ".txt":  (PipelineType.standard, []),
    ".png":  (PipelineType.ocr, [PipelineType.vlm]),
    ".jpg":  (PipelineType.ocr, [PipelineType.vlm]),
    ".jpeg": (PipelineType.ocr, [PipelineType.vlm]),
    ".tiff": (PipelineType.ocr, [PipelineType.vlm]),
    ".bmp":  (PipelineType.ocr, [PipelineType.vlm]),
}

# ---------------------------------------------------------------------------
# Tunable thresholds — change here, propagates everywhere
# ---------------------------------------------------------------------------
_IMAGE_RATIO_FIGURE_HEAVY: float = 0.4   # >40 % of page area covered by images
_TEXT_DENSITY_LOW: float = 200.0         # <200 chars/page → low-text native PDF
_MIN_YOLO_FIGURES_HEAVY: int = 3         # ≥3 YOLO "Picture" detections
_SCORE_MINERU: int = 2                   # score 2–3 → mineru
_SCORE_CONSENSUS: int = 4               # score ≥4 → consensus

_STANDARD_PAGE_SIZE_MARKERS = ("595", "612", "a4", "letter")


class RoutingRules:
    """
    Pure-function routing: ContentProfile → RoutingResult.
    No I/O. No imports from detector.py or any parser. Fully unit-testable
    by constructing ContentProfile instances directly.
    """

    @staticmethod
    def route(profile: ContentProfile) -> RoutingResult:
        result = RoutingRules._route_non_pdf(profile)
        if result is not None:
            return result

        result = RoutingRules._route_encrypted(profile)
        if result is not None:
            return result

        result = RoutingRules._route_scanned(profile)
        if result is not None:
            return result

        return RoutingRules._route_native_pdf(profile)

    # ------------------------------------------------------------------
    # Gate 1: non-PDF extensions
    # ------------------------------------------------------------------

    @staticmethod
    def _route_non_pdf(profile: ContentProfile) -> RoutingResult | None:
        entry = _NON_PDF_ROUTES.get(profile.extension)
        if entry is None:
            return None
        pipeline, alternatives = entry
        category = "Office/web document" if profile.extension in {".docx", ".pptx", ".html", ".htm", ".md", ".txt"} else "Image file"
        return RoutingResult(
            pipeline=pipeline,
            reason=f"{category} ({profile.extension}): routed to {pipeline.value}",
            confidence=0.95,
            alternatives=list(alternatives),
            profile=profile,
        )

    # ------------------------------------------------------------------
    # Gate 2: encrypted PDF
    # ------------------------------------------------------------------

    @staticmethod
    def _route_encrypted(profile: ContentProfile) -> RoutingResult | None:
        if not profile.encrypted:
            return None
        return RoutingResult(
            pipeline=PipelineType.standard,
            reason="Encrypted PDF: content detection not possible; standard (Docling) may handle partial encryption",
            confidence=0.30,
            alternatives=[PipelineType.ocr],
            profile=profile,
        )

    # ------------------------------------------------------------------
    # Gate 3: scanned PDF
    # ------------------------------------------------------------------

    @staticmethod
    def _route_scanned(profile: ContentProfile) -> RoutingResult | None:
        if not profile.is_scanned:
            return None

        if profile.page_count == 1:
            return RoutingResult(
                pipeline=PipelineType.ocr,
                reason="Scanned single-page PDF: OCR is sufficient; VLM cost not justified",
                confidence=0.85,
                alternatives=[PipelineType.vlm],
                profile=profile,
            )

        page_size_lower = profile.page_size.lower()
        is_standard = any(m in page_size_lower for m in _STANDARD_PAGE_SIZE_MARKERS)

        if is_standard:
            return RoutingResult(
                pipeline=PipelineType.ocr,
                reason=f"Scanned PDF ({profile.page_count} pages, standard page size): OCR with table_structure enabled",
                confidence=0.82,
                alternatives=[PipelineType.vlm],
                profile=profile,
            )

        # Non-standard page size: posters, engineering drawings, unusual formats
        return RoutingResult(
            pipeline=PipelineType.vlm,
            reason=f"Scanned PDF with non-standard page size ({profile.page_size!r}): VLM handles unknown layouts",
            confidence=0.70,
            alternatives=[PipelineType.ocr],
            profile=profile,
        )

    # ------------------------------------------------------------------
    # Gate 4: native PDF — content-complexity routing
    # ------------------------------------------------------------------

    @staticmethod
    def _complexity_score(profile: ContentProfile) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []

        # Formulas (weight 2) — YOLO preferred; heuristic fallback
        formula_detected = False
        if profile.yolo_label_counts is not None:
            formula_detected = profile.yolo_label_counts.get("Formula", 0) > 0
        elif profile.has_formula_heuristic is True:
            formula_detected = True
        if formula_detected:
            score += 2
            reasons.append("formulas")

        # Tables (weight 1) — YOLO preferred; fitz fallback
        table_detected = False
        if profile.yolo_label_counts is not None:
            table_detected = profile.yolo_label_counts.get("Table", 0) > 0
        elif profile.table_count is not None:
            table_detected = profile.table_count > 0
        if table_detected:
            score += 1
            reasons.append("tables")

        # Multi-column layout (weight 1)
        if profile.is_multi_column is True:
            score += 1
            reasons.append("multi-column")

        # Figure-heavy (weight 1) — YOLO preferred; image_ratio fallback
        figure_heavy = False
        if profile.yolo_label_counts is not None:
            figure_heavy = profile.yolo_label_counts.get("Picture", 0) >= _MIN_YOLO_FIGURES_HEAVY
        elif profile.image_ratio is not None:
            figure_heavy = profile.image_ratio > _IMAGE_RATIO_FIGURE_HEAVY
        if figure_heavy:
            score += 1
            reasons.append("figure-heavy" if profile.yolo_label_counts is not None else "image-heavy")

        # Low text density on a native PDF (weight 1)
        if profile.text_density is not None and profile.text_density < _TEXT_DENSITY_LOW:
            score += 1
            reasons.append("low-text-density")

        return score, reasons

    @staticmethod
    def _route_native_pdf(profile: ContentProfile) -> RoutingResult:
        score, detected = RoutingRules._complexity_score(profile)
        features = ", ".join(detected) if detected else "none"

        if profile.detection_level == "none":
            return RoutingResult(
                pipeline=PipelineType.pymupdf,
                reason="Native PDF, no content detection run: defaulting to pymupdf (fastest native PDF parser)",
                confidence=0.60,
                alternatives=[PipelineType.standard, PipelineType.pdfplumber],
                profile=profile,
            )

        if score == 0:
            return RoutingResult(
                pipeline=PipelineType.pymupdf,
                reason=f"Native PDF, no complex features detected: pymupdf is fastest for plain-text documents",
                confidence=0.88,
                alternatives=[PipelineType.standard],
                profile=profile,
            )

        if score >= _SCORE_CONSENSUS:
            return RoutingResult(
                pipeline=PipelineType.consensus,
                reason=f"High-complexity native PDF (score={score}, features: {features}): consensus validates structural elements across multiple parsers",
                confidence=0.80,
                alternatives=[PipelineType.vlm, PipelineType.mineru],
                profile=profile,
            )

        if score >= _SCORE_MINERU:
            return RoutingResult(
                pipeline=PipelineType.mineru,
                reason=f"Medium-complexity native PDF (score={score}, features: {features}): mineru handles formulas, multi-column, and tables",
                confidence=0.78,
                alternatives=[PipelineType.consensus, PipelineType.yolo],
                profile=profile,
            )

        # score == 1: single-feature specialist
        if "formulas" in detected:
            return RoutingResult(
                pipeline=PipelineType.mineru,
                reason=f"Native PDF with formulas (score=1): mineru is the only non-VLM parser with formula support",
                confidence=0.82,
                alternatives=[PipelineType.vlm],
                profile=profile,
            )

        if "tables" in detected:
            return RoutingResult(
                pipeline=PipelineType.pdfplumber,
                reason=f"Native PDF with tables only (score=1): pdfplumber has reliable table extraction at low cost",
                confidence=0.84,
                alternatives=[PipelineType.yolo, PipelineType.mineru],
                profile=profile,
            )

        if "multi-column" in detected:
            return RoutingResult(
                pipeline=PipelineType.yolo,
                reason=f"Native PDF with multi-column layout (score=1): yolo handles layout-aware reading order",
                confidence=0.79,
                alternatives=[PipelineType.mineru],
                profile=profile,
            )

        if "figure-heavy" in detected or "image-heavy" in detected:
            return RoutingResult(
                pipeline=PipelineType.yolo,
                reason=f"Native PDF with significant figure content (score=1): yolo detects and labels figure regions",
                confidence=0.76,
                alternatives=[PipelineType.vlm],
                profile=profile,
            )

        if "low-text-density" in detected:
            # Low text on a native PDF: may contain image-only pages missed by pdftotext probe
            return RoutingResult(
                pipeline=PipelineType.ocr,
                reason=f"Native PDF with low text density (<{_TEXT_DENSITY_LOW:.0f} chars/page): possible image-only regions; OCR applied for safety",
                confidence=0.65,
                alternatives=[PipelineType.pymupdf, PipelineType.vlm],
                profile=profile,
            )

        # Should never reach here if scoring is consistent
        return RoutingResult(
            pipeline=PipelineType.pymupdf,
            reason=f"Native PDF, unhandled score branch (score={score}, features: {features}): fallback to pymupdf",
            confidence=0.55,
            alternatives=[PipelineType.standard],
            profile=profile,
        )
