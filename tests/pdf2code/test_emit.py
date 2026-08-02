from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code.emit import emit_passthrough
from corpuscraft.pdf2code.models import BBox, DocumentExtraction, PageExtraction, PathSegment, VectorDrawing


def test_quad_segment_emits_rect_not_polygon() -> None:
    # Regression: a real payslip PDF's "qu" (Quad) drawings list points in
    # UL, LL, UR, LR order, not a walkable perimeter order. Feeding that
    # straight into an SVG <polygon> draws a bowtie (the closing edges cross
    # diagonally) instead of a rectangle. Bounding-box-as-<rect> sidesteps
    # the point-order ambiguity entirely.
    drawing = VectorDrawing(
        bbox=BBox(354.31, 696.87, 551.32, 762.64),
        segments=[
            PathSegment(
                op="qu",
                points=[(354.31, 696.87), (354.31, 762.64), (551.32, 696.87), (551.32, 762.64)],
            )
        ],
        stroke_color="#000000",
        fill_color=None,
        line_width=1.0,
        page_index=0,
    )
    page = PageExtraction(page_index=0, width=600, height=800, rotation=0, primitives=[drawing])
    document = DocumentExtraction(source_path=Path("fake.pdf"), pages=[page])

    html, _css = emit_passthrough(document)

    assert "<polygon" not in html
    assert '<rect x="0.00" y="0.00" width="197.01" height="65.77"' in html
