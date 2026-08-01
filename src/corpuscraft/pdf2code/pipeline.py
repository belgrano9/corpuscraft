from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code import diff as diff_stage
from corpuscraft.pdf2code import extract as extract_stage
from corpuscraft.pdf2code import render as render_stage
from corpuscraft.pdf2code.emit import emit_passthrough
from corpuscraft.pdf2code.models import BBox, DiffResult, LayoutNode, LayoutTree, PageExtraction
from corpuscraft.pdf2code.serde import dump_diff_result, dump_document_extraction, dump_json, dump_layout_tree


def run_skeleton(pdf_path: Path, out_dir: Path, *, dpi: int = 150) -> list[DiffResult]:
    """Extract -> passthrough emit -> render -> diff, dumping every stage's JSON to out_dir.

    styles.py/layout.py don't exist yet, so this builds a flat pseudo layout
    tree (one leaf per primitive, no grouping) purely to give diff.py's
    per-node path something to walk. Stage 3 replaces _flat_layout_tree.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    document = extract_stage.extract_document(pdf_path, out_dir / "extracted_images")
    dump_document_extraction(document, out_dir / "extraction.json")

    html, css = emit_passthrough(document)
    (out_dir / "emitted.html").write_text(html)
    (out_dir / "emitted.css").write_text(css)

    render_result = render_stage.render_html(html, css, out_dir / "render", dpi=dpi)
    dump_json(render_result, out_dir / "render_result.json")

    original_images = render_stage.rasterize_pdf(pdf_path, out_dir / "original_raster", dpi=dpi)

    results: list[DiffResult] = []
    for page in document.pages:
        tree = _flat_layout_tree(page)
        dump_layout_tree(tree, out_dir / f"layout_page{page.page_index}.json")

        result = diff_stage.diff_page(
            original_image=original_images[page.page_index],
            rendered_image=render_result.page_images[page.page_index],
            layout_tree=tree,
            dpi=dpi,
            out_dir=out_dir / "diff",
        )
        dump_diff_result(result, out_dir / f"diff_page{page.page_index}.json")
        results.append(result)

    return results


def _flat_layout_tree(page: PageExtraction) -> LayoutTree:
    root = LayoutNode(
        id=f"page{page.page_index}",
        kind="page",
        bbox=BBox(0, 0, page.width, page.height),
    )
    for index, primitive in enumerate(page.primitives):
        kind = "line" if primitive.kind == "text" else primitive.kind
        root.children.append(
            LayoutNode(
                id=f"page{page.page_index}/prim{index}",
                kind=kind,
                bbox=primitive.bbox,
                primitive_refs=[index],
            )
        )
    return LayoutTree(page_index=page.page_index, root=root)
