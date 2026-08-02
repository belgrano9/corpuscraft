import sys
from pathlib import Path

import fitz


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: uv run examples/pdf2code/dump_drawings.py <pdf_path> <page_index> [y_min] [y_max]")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    page_index = int(sys.argv[2])
    y_min = float(sys.argv[3]) if len(sys.argv) > 3 else None
    y_max = float(sys.argv[4]) if len(sys.argv) > 4 else None

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    for drawing_index, d in enumerate(page.get_drawings()):
        rect = d.get("rect")
        if y_min is not None and rect is not None and (rect.y1 < y_min or rect.y0 > y_max):
            continue

        print(f"--- drawing {drawing_index} ---")
        print(f"rect: {rect}")
        print(f"color: {d.get('color')}  fill: {d.get('fill')}  width: {d.get('width')}")

        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for item in d.get("items", []):
            print(f"  item: {item}")
            for arg in item[1:]:
                if hasattr(arg, "x0"):  # Rect
                    coords = [(arg.x0, arg.y0), (arg.x1, arg.y1)]
                elif hasattr(arg, "x"):  # Point
                    coords = [(arg.x, arg.y)]
                elif hasattr(arg, "ul"):  # Quad
                    coords = [(p.x, p.y) for p in arg]
                else:  # e.g. the trailing orientation int on a "re" item
                    continue
                for x, y in coords:
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)

        print(f"  actual point extent: x=[{min_x:.3f}, {max_x:.3f}] y=[{min_y:.3f}, {max_y:.3f}]")
        if rect is not None:
            mismatch = (
                min_x < rect.x0 - 0.01
                or max_x > rect.x1 + 0.01
                or min_y < rect.y0 - 0.01
                or max_y > rect.y1 + 0.01
            )
            print(f"  MISMATCH with reported rect: {mismatch}")

    doc.close()


if __name__ == "__main__":
    main()
