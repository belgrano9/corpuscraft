from __future__ import annotations

from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from loguru import logger

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

_RENDER_DPI = 150
_SCALE = _RENDER_DPI / 72.0  # pixels per PDF point

_SKIP_LABELS = {"Page-header", "Page-footer"}
_HEADING_LABELS = {"Title": "#", "Section-header": "##"}
_BLOCK_LABELS = {"Table": "[TABLE]", "Picture": "[FIGURE]"}

# DocStructBench (DocLayNet) class index → label
_NAMES: dict[int, str] = {
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


class YoloParser(BaseParser):
    """Layout-aware PDF parser: DocLayout-YOLO detects regions, pdfplumber extracts text.

    Limitations:
    - Native PDFs only — scanned pages yield no text; use the ocr pipeline instead.
    - Reading order is naive (top→bottom, left→right); 2-column layouts interleave.
    """

    def __init__(self, config: ParserConfig) -> None:
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as e:
            raise ImportError(
                "Install the yolo extras: uv add 'corpuscraft[yolo]'"
            ) from e

        self._conf = config.yolo_confidence
        logger.info(f"Loading DocLayout-YOLO: {config.yolo_model}")
        self._model = YOLOv10.from_pretrained(config.yolo_model)

    def parse_file(self, path: Path) -> ParsedDocument:
        pdf_doc = pdfium.PdfDocument(str(path))
        num_pages = len(pdf_doc)
        page_texts: list[str] = []

        with pdfplumber.open(path) as pdf:
            for idx in range(num_pages):
                plumber_page = pdf.pages[idx]

                bitmap = pdf_doc[idx].render(scale=_SCALE)
                pil_image = bitmap.to_pil()

                results = self._model.predict(
                    pil_image, imgsz=1024, conf=self._conf, device="cpu", verbose=False
                )

                boxes = results[0].boxes if results else None
                if boxes is None or len(boxes) == 0:
                    text = plumber_page.extract_text() or ""
                    if text.strip():
                        page_texts.append(text.strip())
                    else:
                        logger.warning(
                            f"Page {idx + 1} of {path.name}: no detections and no "
                            "extractable text — scanned PDF? Use the ocr pipeline."
                        )
                    continue

                detections = sorted(
                    zip(boxes.cls.tolist(), boxes.xyxy.tolist()),
                    key=lambda d: (d[1][1], d[1][0]),  # y1, x1
                )

                segments: list[str] = []
                for cls_id, (x1, y1, x2, y2) in detections:
                    label = _NAMES.get(int(cls_id), "Text")
                    if label in _SKIP_LABELS:
                        continue

                    bbox = (x1 / _SCALE, y1 / _SCALE, x2 / _SCALE, y2 / _SCALE)
                    text = plumber_page.crop(bbox).extract_text() or ""
                    if not text.strip():
                        continue

                    if label in _HEADING_LABELS:
                        segments.append(f"{_HEADING_LABELS[label]} {text.strip()}")
                    elif label in _BLOCK_LABELS:
                        segments.append(f"{_BLOCK_LABELS[label]}\n{text.strip()}")
                    else:
                        segments.append(text.strip())

                if segments:
                    page_texts.append("\n\n".join(segments))

        return ParsedDocument(
            content="\n\n---\n\n".join(page_texts),
            source_path=path,
            pipeline="yolo",
            metadata={
                "file_size": path.stat().st_size,
                "num_pages": num_pages,
            },
        )
