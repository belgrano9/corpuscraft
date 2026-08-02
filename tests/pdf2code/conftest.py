from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from PIL import Image


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A tiny synthetic PDF covering all primitive kinds (text/drawing/image).

    Built on the fly with fitz rather than committed as a binary fixture,
    since *.pdf is gitignored repo-wide.
    """
    img_path = tmp_path / "swatch.png"
    Image.new("RGB", (40, 20), (0, 128, 255)).save(img_path)

    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((20, 40), "Hello World", fontsize=14, fontname="hebo", color=(1, 0, 0))
    page.draw_rect(fitz.Rect(10, 60, 150, 100), color=(0, 0, 1), fill=(0.9, 0.9, 0.9), width=1.5)
    page.draw_line(fitz.Point(10, 120), fitz.Point(150, 120), color=(0, 0.6, 0), width=1)
    page.insert_image(fitz.Rect(160, 60, 200, 80), filename=str(img_path))

    pdf_path = tmp_path / "sample.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def layout_sample_pdf(tmp_path: Path) -> Path:
    """A synthetic PDF with a clear paragraph gap and a bordered 2x2 table.

    Exercises layout.py's whitespace XY-cut (the gap between the two
    paragraphs) and its find_tables() pre-pass (the bordered grid).
    """
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)

    page.insert_text((20, 30), "Title heading", fontsize=12, fontname="hebo")
    page.insert_text((20, 60), "First paragraph line one.", fontsize=9, fontname="helv")
    page.insert_text((20, 72), "First paragraph line two.", fontsize=9, fontname="helv")
    # a wide gap before the next block, well above min_row_gap_pt's default
    page.insert_text((20, 150), "Second paragraph after a gap.", fontsize=9, fontname="helv")

    page.draw_rect(fitz.Rect(20, 200, 220, 240), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(20, 220), fitz.Point(220, 220), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(120, 200), fitz.Point(120, 240), color=(0, 0, 0), width=1)
    page.insert_text((25, 215), "A1", fontsize=8)
    page.insert_text((125, 215), "B1", fontsize=8)
    page.insert_text((25, 235), "A2", fontsize=8)
    page.insert_text((125, 235), "B2", fontsize=8)

    pdf_path = tmp_path / "layout_sample.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path
