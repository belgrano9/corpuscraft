from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code.extract import extract_document
from corpuscraft.pdf2code.layout import build_layout_tree
from corpuscraft.pdf2code.models import LayoutNode


def _collect_refs(node: LayoutNode) -> list[int]:
    refs = list(node.primitive_refs)
    for child in node.children:
        refs.extend(_collect_refs(child))
    return refs


def test_build_layout_tree_detects_table(layout_sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(layout_sample_pdf, tmp_path / "images")
    page = document.pages[0]

    tree = build_layout_tree(page, layout_sample_pdf)

    table_nodes = [c for c in tree.root.children if c.kind == "table"]
    assert len(table_nodes) == 1
    table = table_nodes[0]
    assert len(table.children) == 2  # 2 rows
    assert all(row.kind == "row" for row in table.children)
    assert all(len(row.children) == 2 for row in table.children)  # 2 cols
    assert all(cell.kind == "cell" for row in table.children for cell in row.children)


def test_build_layout_tree_splits_paragraph_gap(layout_sample_pdf: Path, tmp_path: Path) -> None:
    document = extract_document(layout_sample_pdf, tmp_path / "images")
    page = document.pages[0]

    tree = build_layout_tree(page, layout_sample_pdf, min_row_gap_pt=4.0)

    content_node = next(c for c in tree.root.children if c.kind != "table")
    # the 60pt gap before "Second paragraph after a gap." should force a
    # top-level block split rather than lumping everything into one block
    assert content_node.kind == "block"
    assert len(content_node.children) >= 2


def test_build_layout_tree_covers_every_primitive_exactly_once(
    layout_sample_pdf: Path, tmp_path: Path
) -> None:
    document = extract_document(layout_sample_pdf, tmp_path / "images")
    page = document.pages[0]

    tree = build_layout_tree(page, layout_sample_pdf)

    all_refs = _collect_refs(tree.root)
    assert sorted(all_refs) == list(range(len(page.primitives)))
    assert len(all_refs) == len(set(all_refs))
