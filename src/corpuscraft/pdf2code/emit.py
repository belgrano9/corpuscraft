from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from corpuscraft.pdf2code.models import (
    BBox,
    DocumentExtraction,
    ImagePrimitive,
    StyleSheet,
    TextSpan,
    VectorDrawing,
)
from corpuscraft.pdf2code.styles import render_stylesheet_css


def emit_passthrough(document: DocumentExtraction, stylesheet: StyleSheet | None = None) -> tuple[str, str]:
    """Stage-4 stand-in: one absolutely-positioned element per primitive.

    Ignores layout.py's structure (not consumed yet) so extract -> render ->
    diff can run end to end. Drawings/images are stacked behind text (a
    fixed heuristic, not the original z-order) since backgrounds-behind-text
    is the common case and get_drawings() sequence numbers aren't comparable
    to text draw order. The real emit.py replaces this with a grammar
    -constrained DOM built from the layout tree and style classes.

    stylesheet is optional: when given, text spans reference styles.py's
    generated CSS classes (real font resolution/embedding) instead of
    quoting the raw PDF font name directly.
    """
    html = Element("html")
    body = SubElement(html, "body", {"style": "margin:0;padding:0;"})

    for page_num, page in enumerate(document.pages):
        page_style = (
            f"position:relative;width:{page.width:.2f}pt;height:{page.height:.2f}pt;overflow:hidden;"
        )
        if page_num < len(document.pages) - 1:
            page_style += "page-break-after:always;"
        page_div = SubElement(body, "div", {"class": "pdf-page", "style": page_style})

        drawings, images, texts = [], [], []
        for index, primitive in enumerate(page.primitives):
            if isinstance(primitive, VectorDrawing):
                drawings.append((index, primitive))
            elif isinstance(primitive, ImagePrimitive):
                images.append((index, primitive))
            elif isinstance(primitive, TextSpan):
                texts.append((index, primitive))

        for index, drawing in drawings:
            _emit_drawing(page_div, drawing, index)
        for index, image in images:
            _emit_image(page_div, image, index)
        for index, span in texts:
            _emit_text(page_div, span, index, stylesheet, page_num)

    first_page = document.pages[0] if document.pages else None
    page_size_rule = (
        f"@page {{ margin: 0; size: {first_page.width:.2f}pt {first_page.height:.2f}pt; }}"
        if first_page
        else "@page { margin: 0; }"
    )
    css = page_size_rule + "\nbody { margin: 0; }"
    if stylesheet is not None:
        css += "\n" + render_stylesheet_css(stylesheet)
    html_str = "<!doctype html>\n" + tostring(html, encoding="unicode")
    return html_str, css


def _abs_style(bbox: BBox, extra: str = "") -> str:
    return (
        f"position:absolute;left:{bbox.x0:.2f}pt;top:{bbox.y0:.2f}pt;"
        f"width:{bbox.width:.2f}pt;height:{bbox.height:.2f}pt;{extra}"
    )


def _css_font_family(name: str) -> str:
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}", sans-serif'


def _emit_text(
    parent: Element, span: TextSpan, index: int, stylesheet: StyleSheet | None, page_index: int
) -> None:
    class_name = stylesheet.assignments.get(f"{page_index}:{index}") if stylesheet is not None else None
    if class_name is not None:
        css_class = f"primitive text {class_name}"
        style = _abs_style(span.bbox, f"margin:0;white-space:pre;line-height:{span.bbox.height:.2f}pt;")
    else:
        css_class = "primitive text"
        style = _abs_style(
            span.bbox,
            f"margin:0;white-space:pre;font-size:{span.font_size:.2f}pt;"
            f"font-weight:{span.font_weight};font-style:{'italic' if span.italic else 'normal'};"
            f"color:{span.color};font-family:{_css_font_family(span.font_family)};"
            f"line-height:{span.bbox.height:.2f}pt;",
        )
    div = SubElement(parent, "div", {"class": css_class, "data-index": str(index), "style": style})
    div.text = span.text


def _emit_image(parent: Element, image: ImagePrimitive, index: int) -> None:
    style = _abs_style(image.bbox, "object-fit:fill;")
    SubElement(
        parent,
        "img",
        {
            "class": "primitive image",
            "data-index": str(index),
            "style": style,
            "src": image.image_path.resolve().as_uri(),
        },
    )


def _emit_drawing(parent: Element, drawing: VectorDrawing, index: int) -> None:
    bbox = drawing.bbox
    svg = SubElement(
        parent,
        "svg",
        {
            "class": "primitive drawing",
            "data-index": str(index),
            "style": _abs_style(bbox),
            "viewBox": f"0 0 {bbox.width:.2f} {bbox.height:.2f}",
            "xmlns": "http://www.w3.org/2000/svg",
        },
    )
    stroke = drawing.stroke_color or "none"
    fill = drawing.fill_color or "none"
    width = f"{drawing.line_width:.2f}"
    for segment in drawing.segments:
        pts = [(x - bbox.x0, y - bbox.y0) for x, y in segment.points]
        if segment.op == "re" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            SubElement(
                svg,
                "rect",
                {
                    "x": f"{min(x0, x1):.2f}",
                    "y": f"{min(y0, y1):.2f}",
                    "width": f"{abs(x1 - x0):.2f}",
                    "height": f"{abs(y1 - y0):.2f}",
                    "stroke": stroke,
                    "fill": fill,
                    "stroke-width": width,
                },
            )
        elif segment.op == "qu" and len(pts) >= 3:
            # Quad point order isn't a reliable perimeter walk across PDF
            # generators (seen in practice as UL,LL,UR,LR rather than a
            # walkable UL,UR,LR,LL) -- connecting them in raw order as a
            # <polygon> draws a bowtie instead of a rectangle. Nearly all
            # real-world "qu" usage (table cells/borders) is axis-aligned
            # anyway, so use the bounding box instead of trusting order.
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            SubElement(
                svg,
                "rect",
                {
                    "x": f"{min(xs):.2f}",
                    "y": f"{min(ys):.2f}",
                    "width": f"{max(xs) - min(xs):.2f}",
                    "height": f"{max(ys) - min(ys):.2f}",
                    "stroke": stroke,
                    "fill": fill,
                    "stroke-width": width,
                },
            )
        elif segment.op == "l" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            SubElement(
                svg,
                "line",
                {
                    "x1": f"{x0:.2f}",
                    "y1": f"{y0:.2f}",
                    "x2": f"{x1:.2f}",
                    "y2": f"{y1:.2f}",
                    "stroke": stroke,
                    "stroke-width": width,
                },
            )
        elif segment.op == "c" and len(pts) == 4:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
            path = f"M {x0:.2f},{y0:.2f} C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}"
            SubElement(
                svg, "path", {"d": path, "stroke": stroke, "fill": "none", "stroke-width": width}
            )
