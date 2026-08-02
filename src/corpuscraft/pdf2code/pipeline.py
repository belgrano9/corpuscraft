from __future__ import annotations

from pathlib import Path

from corpuscraft.pdf2code import diff as diff_stage
from corpuscraft.pdf2code import extract as extract_stage
from corpuscraft.pdf2code import render as render_stage
from corpuscraft.pdf2code.emit import emit_passthrough
from corpuscraft.pdf2code.layout import build_layout_tree
from corpuscraft.pdf2code.models import DiffResult
from corpuscraft.pdf2code.serde import dump_diff_result, dump_document_extraction, dump_json, dump_layout_tree


def run_skeleton(pdf_path: Path, out_dir: Path, *, dpi: int = 150) -> list[DiffResult]:
    """Extract -> passthrough emit -> render -> diff, dumping every stage's JSON to out_dir.

    styles.py doesn't exist yet, so this still uses the trivial passthrough
    emitter (ignoring the real layout tree's structure/style_class) — only
    diff.py's per-node path benefits from build_layout_tree for now.
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
        tree = build_layout_tree(page, pdf_path)
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
