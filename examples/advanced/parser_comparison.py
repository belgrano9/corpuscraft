"""
Parser Comparison: Docling vs DocLayout-YOLO

For each page, renders a side-by-side PNG showing Tables and Figures
detected by each pipeline on the same rendered page image.

Left panel  — Docling        (Table: blue,   Figure: green)
Right panel — DocLayout-YOLO (Table: red,    Picture: orange)

Usage:
    uv run examples/advanced/parser_comparison.py

Output:
    scratch/comparison/page-001.png, page-002.png, ...
"""

import logging
import time
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
from PIL import Image, ImageDraw

_YOLO_RENDER_SCALE = 150 / 72.0  # matches YoloParser exactly

SOURCE_FILE = Path("data") / "raw" / "bulletin-de-paie-du-011025-au-311025.pdf"
OUTPUT_DIR = Path("scratch") / "comparison"
IMAGE_SCALE = 2.0
YOLO_CONF = 0.2
YOLO_MODEL = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained"

_YOLO_NAMES: dict[int, str] = {
    0: "Caption", 1: "Footnote", 2: "Formula", 3: "List-item",
    4: "Page-footer", 5: "Page-header", 6: "Picture",
    7: "Section-header", 8: "Table", 9: "Text", 10: "Title",
}

_DOCLING_COLORS = {"Table": (30, 100, 255), "Figure": (30, 200, 80)}
_YOLO_COLORS    = {"Table": (220, 50, 50),  "Picture": (255, 140, 0)}

_log = logging.getLogger(__name__)


def _draw_panel(base: Image.Image, boxes: list[dict], title: str) -> Image.Image:
    img = base.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=b["color"], width=3)
        tag = b["label"] + (f" {b['conf']:.2f}" if "conf" in b else "")
        tx, ty = int(b["x1"]), max(0, int(b["y1"]) - 16)
        draw.rectangle([tx, ty, tx + len(tag) * 7, ty + 16], fill=b["color"])
        draw.text((tx + 2, ty + 1), tag, fill=(255, 255, 255))

    header = Image.new("RGB", (img.width, 28), (50, 50, 50))
    ImageDraw.Draw(header).text((8, 6), title, fill=(255, 255, 255))
    panel = Image.new("RGB", (img.width, img.height + 28))
    panel.paste(header, (0, 0))
    panel.paste(img, (0, 28))
    return panel


def _side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 4
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, h), (180, 180, 180))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("Install yolo extras: uv add 'corpuscraft[yolo]'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Docling ──────────────────────────────────────────────────────────────
    _log.info("Running Docling ...")
    opts = PdfPipelineOptions()
    opts.images_scale = IMAGE_SCALE
    opts.generate_page_images = True
    opts.generate_picture_images = True

    conv_res = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    ).convert(str(SOURCE_FILE))
    doc = conv_res.document

    docling_boxes: dict[int, list[dict]] = {}
    for element, _ in doc.iterate_items():
        if not isinstance(element, (TableItem, PictureItem)):
            continue
        label = "Table" if isinstance(element, TableItem) else "Figure"
        color = _DOCLING_COLORS[label]
        for prov in element.prov:
            pno = prov.page_no
            pil = doc.pages[pno].image.pil_image
            ph_pts = pil.height / IMAGE_SCALE  # page height in PDF points
            bx = prov.bbox                     # PDF coords: origin bottom-left
            docling_boxes.setdefault(pno, []).append({
                "label": label, "color": color,
                "x1": bx.l * IMAGE_SCALE,
                "y1": (ph_pts - bx.t) * IMAGE_SCALE,
                "x2": bx.r * IMAGE_SCALE,
                "y2": (ph_pts - bx.b) * IMAGE_SCALE,
            })

    n_docling = sum(len(v) for v in docling_boxes.values())
    _log.info(f"  Docling: {n_docling} detection(s)")

    # ── YOLO — render with pypdfium2 exactly as YoloParser does ─────────────
    _log.info("Loading DocLayout-YOLO ...")
    pt = next(f for f in list_repo_files(YOLO_MODEL) if f.endswith(".pt"))
    model = YOLOv10(str(hf_hub_download(repo_id=YOLO_MODEL, filename=pt)))

    pdf_doc = pdfium.PdfDocument(str(SOURCE_FILE))
    yolo_boxes: dict[int, list[dict]] = {}
    yolo_images: dict[int, Image.Image] = {}

    for idx in range(len(pdf_doc)):
        pno = idx + 1  # 1-based to match docling page numbering
        pil = pdf_doc[idx].render(scale=_YOLO_RENDER_SCALE).to_pil().convert("RGB")
        yolo_images[pno] = pil

        results = model.predict(pil, imgsz=1024, conf=YOLO_CONF, device="cpu", verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            continue
        page_elements: list[dict] = []
        page_captions: list[dict] = []
        for cls_id, conf, (x1, y1, x2, y2) in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            lbl = _YOLO_NAMES.get(int(cls_id), "Unknown")
            box = {"label": lbl, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if lbl in ("Table", "Picture"):
                box["color"] = _YOLO_COLORS[lbl]
                page_elements.append(box)
            elif lbl == "Caption":
                page_captions.append(box)

        # Merge each Caption into the nearest Table/Picture by center distance
        for cap in page_captions:
            if not page_elements:
                break
            cx, cy = (cap["x1"] + cap["x2"]) / 2, (cap["y1"] + cap["y2"]) / 2
            nearest = min(page_elements, key=lambda e: (
                ((e["x1"] + e["x2"]) / 2 - cx) ** 2 + ((e["y1"] + e["y2"]) / 2 - cy) ** 2
            ))
            nearest["x1"] = min(nearest["x1"], cap["x1"])
            nearest["y1"] = min(nearest["y1"], cap["y1"])
            nearest["x2"] = max(nearest["x2"], cap["x2"])
            nearest["y2"] = max(nearest["y2"], cap["y2"])

        yolo_boxes.setdefault(pno, []).extend(page_elements)

    n_yolo = sum(len(v) for v in yolo_boxes.values())
    _log.info(f"  YOLO: {n_yolo} detection(s)")

    # ── Render ───────────────────────────────────────────────────────────────
    _log.info("Rendering comparison pages ...")
    for pno, page in doc.pages.items():
        docling_base = page.image.pil_image
        yolo_base = yolo_images.get(pno, docling_base)
        left  = _draw_panel(docling_base, docling_boxes.get(pno, []), f"Docling — page {pno}")
        right = _draw_panel(yolo_base,    yolo_boxes.get(pno, []),    f"DocLayout-YOLO — page {pno}")
        out = _side_by_side(left, right)
        path = OUTPUT_DIR / f"page-{pno:03d}.png"
        out.save(path)
        _log.info(f"  {path}")

    _log.info(f"Done in {time.time() - t0:.1f}s  →  {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
