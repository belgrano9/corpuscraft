from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from corpuscraft.preprocessing.models import PDFMetadata, PreprocessedPDF
from corpuscraft.routing.detector import ContentDetector
from corpuscraft.routing.models import ContentProfile, RoutingResult
from corpuscraft.routing.rules import RoutingRules


class PipelineRouter:
    """
    Facade that combines ContentDetector and RoutingRules into a single call.

    Typical usage::

        router = PipelineRouter(detection_level="basic")

        # After running PopplerPreprocessor (recommended — accurate is_scanned):
        result = router.route(preprocessed)

        # From a bare file path (is_scanned defaults to False):
        result = router.route_path(Path("paper.pdf"))

        print(result.pipeline.value)   # e.g. "mineru"
        print(result.reason)           # human-readable explanation
        print(result.confidence)       # 0.0 – 1.0
        print(result.alternatives)     # ranked fallback pipelines
    """

    def __init__(
        self,
        detection_level: Literal["none", "basic", "enhanced"] = "basic",
        sample_pages: int = 5,
        yolo_model_id: str = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
        yolo_confidence: float = 0.2,
    ) -> None:
        self._detector = ContentDetector(
            level=detection_level,
            sample_pages=sample_pages,
            yolo_model_id=yolo_model_id,
            yolo_confidence=yolo_confidence,
        )

    def route(self, preprocessed: PreprocessedPDF) -> RoutingResult:
        """
        Primary routing path. Requires a PreprocessedPDF from PopplerPreprocessor
        so that is_scanned and metadata are accurate.
        """
        profile = self._detector.detect(preprocessed)
        result = RoutingRules.route(profile)
        logger.debug(
            "route({name}): {pipeline} (confidence={conf:.2f}) — {reason}",
            name=preprocessed.source_path.name,
            pipeline=result.pipeline.value,
            conf=result.confidence,
            reason=result.reason,
        )
        return result

    def route_path(
        self,
        path: Path,
        *,
        is_scanned_hint: bool = False,
    ) -> RoutingResult:
        """
        Convenience method that works directly from a file path.

        For non-PDF files: pure extension routing, zero I/O.
        For PDFs: reads basic metadata via fitz (page count, encryption) but
        does NOT run poppler, so is_scanned defaults to False unless the
        caller passes is_scanned_hint=True. Prefer route() when you have a
        PreprocessedPDF for accurate results.
        """
        ext = path.suffix.lower()

        if ext != ".pdf":
            profile = ContentProfile(
                extension=ext,
                page_count=1,
                is_scanned=False,
                encrypted=False,
                page_size="",
                detection_level="none",
            )
            return RoutingRules.route(profile)

        # PDF: read basic metadata via fitz
        encrypted = False
        page_count = 1
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(str(path))
            encrypted = doc.is_encrypted
            page_count = doc.page_count
            doc.close()
        except Exception:
            pass

        metadata = PDFMetadata(
            page_count=page_count,
            encrypted=encrypted,
            pdf_version="",
            title="",
            author="",
            page_size="",
            raw={},
        )
        synthetic = PreprocessedPDF(
            source_path=path,
            output_dir=path.parent,
            metadata=metadata,
            is_scanned=is_scanned_hint,
        )
        return self.route(synthetic)
