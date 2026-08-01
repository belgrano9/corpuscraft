from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code.emit import emit_passthrough
from corpuscraft.pdf2code.extract import extract_document
from corpuscraft.pdf2code.render import rasterize_pdf, render_html


def test_render_html_produces_pdf_and_rasters(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")
    html, css = emit_passthrough(document)

    result = render_html(html, css, tmp_path / "render", dpi=100)

    assert result.pdf_path.exists()
    assert result.dpi == 100
    assert len(result.page_images) == 1
    assert result.page_images[0].exists()


def test_rasterize_pdf_matches_page_count(sample_pdf: Path, tmp_path: Path) -> None:
    pages = rasterize_pdf(sample_pdf, tmp_path / "raster", dpi=100)
    assert len(pages) == 1
    assert pages[0].exists()
