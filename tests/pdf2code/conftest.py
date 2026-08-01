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
