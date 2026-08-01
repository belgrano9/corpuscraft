from pathlib import Path

import fitz

from corpuscraft.pdf2code.pipeline import run_skeleton


def _build_sample_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text(
        (30, 50), "CorpusCraft PDF-to-Code Skeleton", fontsize=16, fontname="hebo", color=(0.1, 0.1, 0.1)
    )
    page.insert_text((30, 90), "This paragraph exercises the walking skeleton:", fontsize=11, fontname="helv")
    page.insert_text((30, 108), "extract -> passthrough emit -> render -> diff.", fontsize=11, fontname="helv")
    page.draw_rect(
        fitz.Rect(30, 130, 370, 180), color=(0, 0, 0.6), fill=(0.85, 0.9, 1.0), width=1.5
    )
    page.insert_text((45, 158), "A bordered, filled box.", fontsize=11, fontname="helv", color=(0, 0, 0.6))
    page.draw_line(fitz.Point(30, 200), fitz.Point(370, 200), color=(0.6, 0, 0), width=1)
    doc.save(path)
    doc.close()


def main() -> None:
    pdf_path = Path("scratch/pdf2code/sample.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _build_sample_pdf(pdf_path)

    out_dir = Path("scratch/pdf2code/run")
    results = run_skeleton(pdf_path, out_dir, dpi=150)

    for result in results:
        print(f"global_score={result.global_score:.3f}  global_mae={result.global_mae:.1f}")
        print("worst nodes:")
        for node in result.node_diffs[:5]:
            print(f"  {node.node_id:24s} score={node.score:.3f} mae={node.mae:.1f} bbox={node.bbox.as_tuple()}")
        print(f"visualization: {result.visualization_path}")

    print(f"\nIntermediate JSON dumped under: {out_dir}")


if __name__ == "__main__":
    main()
