from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from loguru import logger

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

_YOLO_SCALE = 150 / 72.0  # YoloParser render DPI → PDF points


# ── bbox helpers ─────────────────────────────────────────────────────────────

Bbox = tuple[float, float, float, float]  # x1, y1, x2, y2 — top-left origin, PDF pts


def _iom(a: Bbox, b: Bbox) -> float:
    """Intersection over Minimum — handles size-mismatched boxes from different parsers."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


def _to_topleft_pts(
    bbox: tuple,
    *,
    scale: float = 1.0,
    flip_y: bool = False,
    page_h: float = 0.0,
) -> Bbox:
    x1, y1, x2, y2 = (v / scale for v in bbox)
    if flip_y:
        y1, y2 = page_h - y2, page_h - y1
    return (x1, y1, x2, y2)


def _best_iom(candidate: Bbox, others: list[Bbox]) -> float:
    return max((_iom(candidate, o) for o in others), default=0.0)


# ── HTML table → markdown ────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._rows: list[list[str]] = []
        self._cur_row: list[str] = []
        self._cur_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._cur_row.append(" ".join(self._cur_cell).strip())
            self._in_cell = False
        elif tag == "tr" and self._cur_row:
            self._rows.append(self._cur_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cur_cell.append(data.strip())

    def to_markdown(self) -> str:
        if not self._rows:
            return ""
        header = self._rows[0]
        sep = ["---"] * len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in self._rows[1:]:
            # Pad or trim to match header width
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[: len(header)]) + " |")
        return "\n".join(lines)


def _html_to_markdown(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    return p.to_markdown()


# ── per-parser runners (each opens its own file handle) ──────────────────────

def _run_docling(path: Path) -> dict[int, list[Bbox]]:
    """Returns page_no → list of normalised bboxes for Table/Figure detections."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem

    opts = PdfPipelineOptions()
    opts.images_scale = 1.0
    opts.generate_page_images = True
    # Force CPU to avoid OOM when MinerU also holds GPU memory
    import os
    os.environ.setdefault("DOCLING_DEVICE", "cpu")

    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = conv.convert(str(path)).document

    boxes: dict[int, list[Bbox]] = {}
    for element, _ in doc.iterate_items():
        if not isinstance(element, (TableItem, PictureItem)):
            continue
        for prov in element.prov:
            pno = prov.page_no
            page = doc.pages[pno]
            ph_pts = page.image.pil_image.height / 1.0  # scale=1.0
            bx = prov.bbox
            norm = _to_topleft_pts(
                (bx.l, bx.b, bx.r, bx.t), flip_y=True, page_h=ph_pts
            )
            boxes.setdefault(pno, []).append(norm)
    return boxes


def _run_yolo(path: Path, conf: float, model_id: str) -> dict[int, list[Bbox]]:
    """Returns page_no → list of normalised bboxes for Table/Picture detections."""
    import pypdfium2 as pdfium
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download, list_repo_files

    model_path = Path(model_id)
    if not model_path.exists():
        pt_files = [f for f in list_repo_files(model_id) if f.endswith(".pt")]
        model_path = Path(hf_hub_download(repo_id=model_id, filename=pt_files[0]))

    model = YOLOv10(str(model_path))
    pdf_doc = pdfium.PdfDocument(str(path))
    _LABELS = {8: "Table", 6: "Picture"}

    boxes: dict[int, list[Bbox]] = {}
    for idx in range(len(pdf_doc)):
        pno = idx + 1
        pil = pdf_doc[idx].render(scale=_YOLO_SCALE).to_pil()
        results = model.predict(pil, imgsz=1024, conf=conf, device="cpu", verbose=False)
        det = results[0].boxes if results else None
        if det is None or len(det) == 0:
            continue
        for cls_id, (x1, y1, x2, y2) in zip(det.cls.tolist(), det.xyxy.tolist()):
            if int(cls_id) not in _LABELS:
                continue
            norm = _to_topleft_pts((x1, y1, x2, y2), scale=_YOLO_SCALE)
            boxes.setdefault(pno, []).append(norm)
    return boxes


def _run_mineru(path: Path) -> dict[int, list[dict]]:
    """Returns page_no → list of para_blocks (type, bbox, content)."""
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
    from mineru.utils.enum_class import BlockType

    pdf_bytes = path.read_bytes()
    pages: dict[int, list[dict]] = {}

    _STRUCTURAL = {BlockType.TABLE, BlockType.IMAGE, BlockType.CHART}

    class _NullWriter:
        def write(self, p: str, d: bytes) -> None:
            pass

    def on_doc_ready(
        doc_index: int, model_list: list, middle_json: dict, ocr_enable: bool
    ) -> None:
        for page in middle_json.get("pdf_info", []):
            pno = page.get("page_idx", 0) + 1
            for block in page.get("para_blocks", []):
                pages.setdefault(pno, []).append(block)

    doc_analyze_streaming(
        pdf_bytes_list=[pdf_bytes],
        image_writer_list=[_NullWriter()],
        lang_list=[""],
        on_doc_ready=on_doc_ready,
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
    )
    return pages


# ── consensus parser ──────────────────────────────────────────────────────────

class ConsensusParser(BaseParser):
    """Runs Docling, YOLO, and MinerU in parallel and validates structural
    elements (tables, figures) by bounding-box agreement before building
    the final markdown output.

    Text strategy:
    - Body text:  MinerU unconditionally (best reading order).
    - Tables:     MinerU HTML→markdown, only when ≥consensus_min_votes other
                  parsers also detect a table in the same region (IoM ≥ consensus_iom).
    - Figures:    [FIGURE] placeholder when confirmed, skipped otherwise.

    GPU OOM prevention: Docling and YOLO run on CPU; MinerU uses its default
    device (auto-selected based on available VRAM).
    """

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        # Pre-import all heavy modules in the calling thread before spawning
        # the pool — Python's import lock causes deadlocks when multiple threads
        # race to import the same module chain simultaneously.
        from mineru.utils.enum_class import BlockType, MakeMode  # noqa: F401
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming  # noqa: F401
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown
        from docling.document_converter import DocumentConverter  # noqa: F401
        try:
            from doclayout_yolo import YOLOv10  # noqa: F401
        except ImportError:
            pass

        iom_thresh = self.config.consensus_iom
        min_votes = self.config.consensus_min_votes

        # NOTE: PyTorch's global state is not safe across threads during model
        # initialisation. Parsers run sequentially here; true inference parallelism
        # can be added later once models are pre-warmed and a lock layer exists.
        logger.info(f"ConsensusParser: running 3 parsers sequentially on {path.name}")

        docling_boxes: dict[int, list[Bbox]] = _run_docling(path)
        yolo_boxes: dict[int, list[Bbox]] = _run_yolo(
            path, self.config.yolo_confidence, self.config.yolo_model
        )
        mineru_pages: dict[int, list[dict]] = _run_mineru(path)

        logger.info(
            f"  Docling: {sum(len(v) for v in docling_boxes.values())} detections  "
            f"YOLO: {sum(len(v) for v in yolo_boxes.values())} detections"
        )

        _STRUCTURAL = {BlockType.TABLE, BlockType.IMAGE, BlockType.CHART}
        _TEXT_TYPES = {
            BlockType.TEXT, BlockType.TITLE, BlockType.ABSTRACT,
            BlockType.REF_TEXT, BlockType.LIST, BlockType.INDEX,
            BlockType.INTERLINE_EQUATION, BlockType.CODE,
        }

        md_blocks: list[str] = []

        for pno in sorted(mineru_pages):
            d_boxes = docling_boxes.get(pno, [])
            y_boxes = yolo_boxes.get(pno, [])

            for block in mineru_pages[pno]:
                btype = block.get("type", "")
                bbox_raw = block.get("bbox")

                if btype in _TEXT_TYPES:
                    lines = make_blocks_to_markdown([block], MakeMode.NLP_MD)
                    md_blocks.extend(lines)
                    continue

                if btype not in _STRUCTURAL or not bbox_raw:
                    continue

                candidate: Bbox = tuple(bbox_raw)  # type: ignore[assignment]

                votes = 0
                if d_boxes and _best_iom(candidate, d_boxes) >= iom_thresh:
                    votes += 1
                if y_boxes and _best_iom(candidate, y_boxes) >= iom_thresh:
                    votes += 1

                if votes < min_votes:
                    logger.debug(
                        f"  page {pno} {btype} bbox={candidate} — uncertain (votes={votes}), skipping"
                    )
                    continue

                if btype == BlockType.TABLE:
                    html = self._extract_table_html(block)
                    if html:
                        md = _html_to_markdown(html)
                        if md:
                            md_blocks.append(md)
                            continue
                    # No HTML → fall back to pdfplumber text extraction
                    fallback = self._pdfplumber_crop(path, pno, candidate)
                    if fallback:
                        md_blocks.append(fallback)
                else:
                    md_blocks.append("[FIGURE]")

        content = "\n\n".join(b for b in md_blocks if b.strip())
        return ParsedDocument(
            content=content,
            source_path=path,
            pipeline="consensus",
            metadata={
                "file_size": path.stat().st_size,
                "consensus_iom": iom_thresh,
                "consensus_min_votes": min_votes,
            },
        )

    @staticmethod
    def _extract_table_html(block: dict) -> str:
        for sub in block.get("blocks", []):
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    html = span.get("html", "")
                    if html:
                        return html
        return ""

    @staticmethod
    def _pdfplumber_crop(path: Path, page_no: int, bbox: Bbox) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                page = pdf.pages[page_no - 1]
                text = page.crop(bbox).extract_text() or ""
                return text.strip()
        except Exception:
            return ""
