import sys
from pathlib import Path

from corpuscraft.pdf2code.pipeline import run_skeleton


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run examples/pdf2code/run_on_file.py <path-to-pdf> [out_dir] [dpi]")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("scratch/pdf2code") / pdf_path.stem
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 150

    results = run_skeleton(pdf_path, out_dir, dpi=dpi)

    for page_index, result in enumerate(results):
        print(f"page {page_index}: global_score={result.global_score:.3f}  global_mae={result.global_mae:.1f}")
        print("  worst nodes:")
        for node in result.node_diffs[:5]:
            print(f"    {node.node_id:24s} score={node.score:.3f} mae={node.mae:.1f} bbox={node.bbox.as_tuple()}")
        print(f"  visualization: {result.visualization_path}")

    print(f"\nIntermediate JSON dumped under: {out_dir}")


if __name__ == "__main__":
    main()
