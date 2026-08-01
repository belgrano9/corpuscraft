from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code.extract import extract_document
from corpuscraft.pdf2code.models import ImagePrimitive, TextSpan, VectorDrawing
from corpuscraft.pdf2code.serde import dump_document_extraction, load_document_extraction


def test_extract_document_finds_all_primitive_kinds(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")
    assert len(document.pages) == 1
    page = document.pages[0]

    texts = [p for p in page.primitives if isinstance(p, TextSpan)]
    drawings = [p for p in page.primitives if isinstance(p, VectorDrawing)]
    images = [p for p in page.primitives if isinstance(p, ImagePrimitive)]

    assert any("Hello World" in t.text for t in texts)
    assert all(t.font_weight == 700 for t in texts)
    assert len(drawings) >= 2  # rect + line
    assert len(images) == 1
    assert images[0].image_path.exists()


def test_extract_document_json_roundtrip(sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(sample_pdf, tmp_path / "images")

    out = tmp_path / "extraction.json"
    dump_document_extraction(document, out)
    loaded = load_document_extraction(out)

    assert len(loaded.pages) == len(document.pages)
    assert len(loaded.pages[0].primitives) == len(document.pages[0].primitives)
    assert loaded.pages[0].primitives[0].kind == document.pages[0].primitives[0].kind
