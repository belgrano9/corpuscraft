from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from corpuscraft.pdf2code.extract import extract_document
from corpuscraft.pdf2code.serde import dump_stylesheet, load_stylesheet
from corpuscraft.pdf2code.styles import _resolve_font, build_stylesheet, render_stylesheet_css

_DEJAVU_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


@pytest.fixture
def embedded_font_pdf(tmp_path: Path) -> Path:
    """A PDF with one base-14 span and one span using a genuinely embedded
    font, so font resolution can be tested against both code paths.
    """
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    if _DEJAVU_PATH.exists():
        page.insert_font(fontname="DejaVu", fontfile=str(_DEJAVU_PATH))
        page.insert_text((20, 40), "Embedded font text", fontsize=12, fontname="DejaVu")
    page.insert_text((20, 80), "Base14 text", fontsize=12, fontname="helv")
    pdf_path = tmp_path / "fonts.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_build_stylesheet_clusters_by_style_key(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")

    stylesheet = build_stylesheet(document, tmp_path / "fonts")

    assert stylesheet.classes, "expected at least one style class"
    assert len(stylesheet.assignments) == 1  # sample_pdf has exactly one text span
    class_name = next(iter(stylesheet.assignments.values()))
    assert any(c.name == class_name for c in stylesheet.classes)


@pytest.mark.skipif(not _DEJAVU_PATH.exists(), reason="DejaVuSans.ttf not available on this machine")
def test_build_stylesheet_extracts_embedded_font(embedded_font_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(embedded_font_pdf, tmp_path / "images")

    stylesheet = build_stylesheet(document, tmp_path / "fonts")

    embedded_faces = [f for f in stylesheet.font_faces if f.font_path is not None]
    assert len(embedded_faces) == 1
    assert embedded_faces[0].font_path.exists()
    assert embedded_faces[0].font_path.read_bytes()  # non-empty font file


def test_build_stylesheet_falls_back_for_base14_font(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")

    stylesheet = build_stylesheet(document, tmp_path / "fonts")

    # sample_pdf's only span uses a base-14 font (no embedded program) --
    # resolution must not crash, and must produce no bogus @font-face rule.
    assert stylesheet.font_faces == []


def test_render_stylesheet_css_produces_font_face_and_class_rules(
    embedded_font_pdf: Path, tmp_path: Path
) -> None:
    document = extract_document(embedded_font_pdf, tmp_path / "images")
    stylesheet = build_stylesheet(document, tmp_path / "fonts")

    css = render_stylesheet_css(stylesheet)

    for style_class in stylesheet.classes:
        assert f".{style_class.name} {{" in css
    for face in stylesheet.font_faces:
        assert "@font-face" in css
        assert face.font_path.resolve().as_uri() in css


class _StubDoc:
    """Fakes fitz.Document.extract_font() with fixed per-xref byte buffers."""

    def __init__(self, buffers: dict[int, bytes]) -> None:
        self._buffers = buffers

    def extract_font(self, xref: int) -> tuple[str, str, str, bytes]:
        return ("stub", "ttf", "TrueType", self._buffers.get(xref, b""))


def test_resolve_font_prefers_exact_match_over_longer_substring(tmp_path: Path) -> None:
    # Regression: "Verdana" and "Verdana-Bold" both normalize such that
    # "verdana" is a substring of "verdanabold". The old length-sorted
    # heuristic picked the longer (but wrong) candidate even when an exact
    # match existed, collapsing regular and bold onto the same font file.
    font_resources = [("verdana", 1), ("verdanabold", 2)]
    doc = _StubDoc({1: b"regular-bytes", 2: b"bold-bytes"})

    face = _resolve_font("Verdana", font_resources, doc, tmp_path)

    assert face.font_path is not None
    assert face.font_path.read_bytes() == b"regular-bytes"


def test_stylesheet_json_roundtrip(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")
    stylesheet = build_stylesheet(document, tmp_path / "fonts")

    out = tmp_path / "stylesheet.json"
    dump_stylesheet(stylesheet, out)
    loaded = load_stylesheet(out)

    assert len(loaded.classes) == len(stylesheet.classes)
    assert loaded.assignments == stylesheet.assignments
