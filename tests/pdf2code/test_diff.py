from __future__ import annotations

from pathlib import Path

import numpy as np

from corpuscraft.pdf2code.diff import _crop, diff_page
from corpuscraft.pdf2code.emit import emit_passthrough
from corpuscraft.pdf2code.extract import extract_document
from corpuscraft.pdf2code.models import BBox
from corpuscraft.pdf2code.pipeline import _flat_layout_tree
from corpuscraft.pdf2code.render import rasterize_pdf, render_html


def test_diff_page_produces_ranked_node_diffs(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")
    html, css = emit_passthrough(document)
    render_result = render_html(html, css, tmp_path / "render", dpi=100)
    original_images = rasterize_pdf(sample_pdf, tmp_path / "original", dpi=100)

    page = document.pages[0]
    tree = _flat_layout_tree(page)

    result = diff_page(
        original_image=original_images[0],
        rendered_image=render_result.page_images[0],
        layout_tree=tree,
        dpi=100,
        out_dir=tmp_path / "diff",
    )

    assert -1.0 <= result.global_score <= 1.0
    assert result.global_mae >= 0
    assert result.node_diffs, "expected at least one leaf node diff"
    assert result.visualization_path is not None
    assert result.visualization_path.exists()

    scores = [n.score for n in result.node_diffs]
    assert scores == sorted(scores)  # worst (least similar) first


def test_crop_zero_height_bbox_is_empty() -> None:
    # Regression: a dashed/dotted rule segment can have y0 == y1 exactly.
    # floor(y0*scale) and round(y1*scale) can land on different pixel rows
    # (e.g. 587.0pt @ 150dpi -> 1223.958 -> floor=1223, round=1224),
    # fabricating a spurious 1px-tall crop out of zero actual area.
    arr = np.zeros((1754, 1240))
    bbox = BBox(x0=359.83, y0=587.0, x1=360.40, y1=587.0)
    assert _crop(arr, bbox, dpi=150).size == 0
