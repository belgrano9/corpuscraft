from __future__ import annotations

from pathlib import Path

import fitz

from corpuscraft.pdf2code.models import (
    BBox,
    DocumentExtraction,
    ImagePrimitive,
    PageExtraction,
    PathSegment,
    Primitive,
    TextSpan,
    VectorDrawing,
)

_ITALIC_FLAG = 1 << 1
_BOLD_FLAG = 1 << 4


def extract_document(pdf_path: Path, image_dir: Path) -> DocumentExtraction:
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        pages = [_extract_page(doc, page_index, image_dir) for page_index in range(doc.page_count)]
    finally:
        doc.close()
    return DocumentExtraction(source_path=pdf_path, pages=pages)


def _extract_page(doc: fitz.Document, page_index: int, image_dir: Path) -> PageExtraction:
    page = doc[page_index]
    primitives: list[Primitive] = []
    primitives.extend(_extract_text_spans(page, page_index))
    primitives.extend(_extract_drawings(page, page_index))
    primitives.extend(_extract_images(doc, page, page_index, image_dir))
    return PageExtraction(
        page_index=page_index,
        width=page.rect.width,
        height=page.rect.height,
        rotation=page.rotation,
        primitives=primitives,
    )


def _extract_text_spans(page: fitz.Page, page_index: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks, handled separately
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if not span.get("text", "").strip():
                    continue
                bbox = BBox(*span["bbox"])
                flags = span.get("flags", 0)
                font_name = span.get("font", "")
                bold = bool(flags & _BOLD_FLAG) or "bold" in font_name.lower()
                italic = bool(flags & _ITALIC_FLAG) or "italic" in font_name.lower() or "oblique" in font_name.lower()
                color_int = span.get("color", 0)
                origin = span.get("origin", (bbox.x0, bbox.y1))
                spans.append(
                    TextSpan(
                        text=span["text"],
                        bbox=bbox,
                        font_family=font_name,
                        font_size=span.get("size", 0.0),
                        font_weight=700 if bold else 400,
                        italic=italic,
                        color=f"#{color_int:06x}",
                        origin=(origin[0], origin[1]),
                        page_index=page_index,
                    )
                )
    return spans


def _extract_drawings(page: fitz.Page, page_index: int) -> list[VectorDrawing]:
    drawings: list[VectorDrawing] = []
    for d in page.get_drawings():
        rect = d.get("rect")
        # a zero-width/height line is a legitimate degenerate rect (fitz
        # flags it is_empty) — only skip when there's no rect at all.
        if rect is None:
            continue
        segments = [_path_segment(item) for item in d.get("items", [])]
        drawings.append(
            VectorDrawing(
                bbox=BBox(rect.x0, rect.y0, rect.x1, rect.y1),
                segments=segments,
                stroke_color=_color_to_hex(d.get("color")),
                fill_color=_color_to_hex(d.get("fill")),
                line_width=d.get("width") or 0.0,
                page_index=page_index,
            )
        )
    return drawings


def _path_segment(item: tuple) -> PathSegment:
    op = item[0]
    if op == "re":
        rect = item[1]
        points = [(rect.x0, rect.y0), (rect.x1, rect.y1)]
    elif op == "qu":
        points = [(pt.x, pt.y) for pt in item[1]]
    else:  # "l" (line) and "c" (cubic bezier) both carry a run of Points
        points = [(pt.x, pt.y) for pt in item[1:]]
    return PathSegment(op=op, points=points)


def _color_to_hex(color: tuple[float, ...] | None) -> str | None:
    if not color:
        return None
    r, g, b = (tuple(color) + (0.0, 0.0, 0.0))[:3]
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _extract_images(
    doc: fitz.Document, page: fitz.Page, page_index: int, image_dir: Path
) -> list[ImagePrimitive]:
    images: list[ImagePrimitive] = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:  # CMYK/other -> RGB(A)
            pix = fitz.Pixmap(fitz.csRGB, pix)
        out_path = image_dir / f"page{page_index}_img{img_index}_{xref}.png"
        pix.save(out_path)
        for rect in rects:
            images.append(
                ImagePrimitive(
                    bbox=BBox(rect.x0, rect.y0, rect.x1, rect.y1),
                    image_path=out_path,
                    xref=xref,
                    width_px=pix.width,
                    height_px=pix.height,
                    rotation=page.rotation,
                    page_index=page_index,
                )
            )
    return images
