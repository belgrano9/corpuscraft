from __future__ import annotations

from pathlib import Path

import fitz
from weasyprint import CSS, HTML

from corpuscraft.pdf2code.models import RenderResult


def render_html(
    html: str, css: str, out_dir: Path, *, base_url: str | None = None, dpi: int = 150
) -> RenderResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "rendered.pdf"
    HTML(string=html, base_url=base_url).write_pdf(pdf_path, stylesheets=[CSS(string=css)])
    page_images = rasterize_pdf(pdf_path, out_dir, dpi=dpi)
    return RenderResult(pdf_path=pdf_path, page_images=page_images, dpi=dpi)


def rasterize_pdf(pdf_path: Path, out_dir: Path, *, dpi: int = 150, prefix: str = "page") -> list[Path]:
    """Rasterize every page of pdf_path to PNG via PyMuPDF.

    Used for both the rendered output and the original source PDF so both
    sides of diff.py's comparison go through the identical renderer/DPI —
    no separate tool (e.g. poppler) that could introduce AA/rounding drift.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(pdf_path)
    paths: list[Path] = []
    try:
        for page_index in range(doc.page_count):
            pix = doc[page_index].get_pixmap(matrix=matrix)
            out_path = out_dir / f"{prefix}{page_index}.png"
            pix.save(out_path)
            paths.append(out_path)
    finally:
        doc.close()
    return paths
