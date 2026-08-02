import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run examples/pdf2code/probe_tables.py <pdf_path> [page_index] [dpi]")
        raise SystemExit(1)

    pdf_path = Path(sys.argv[1])
    page_filter = int(sys.argv[2]) if len(sys.argv) > 2 else None
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    scale = dpi / 72

    out_dir = Path("scratch/pdf2code") / f"{pdf_path.stem}_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page_indices = [page_filter] if page_filter is not None else range(doc.page_count)

    for page_index in page_indices:
        page = doc[page_index]
        finder = page.find_tables()
        print(f"page {page_index}: {len(finder.tables)} table(s) found")

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        raw_path = out_dir / f"page{page_index}_raw.png"
        pix.save(raw_path)

        image = Image.open(raw_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        for table_index, table in enumerate(finder.tables):
            x0, y0, x1, y1 = (v * scale for v in table.bbox)
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
            print(f"  table {table_index}: bbox={table.bbox} rows={table.row_count} cols={table.col_count}")
            for row in table.extract():
                print(f"    {row}")
            for cell in table.cells:
                if cell is None:
                    continue
                cx0, cy0, cx1, cy1 = (v * scale for v in cell)
                draw.rectangle([cx0, cy0, cx1, cy1], outline=(0, 120, 255), width=1)

        overlay_path = out_dir / f"page{page_index}_tables.png"
        image.save(overlay_path)
        print(f"  overlay: {overlay_path}")

    doc.close()


if __name__ == "__main__":
    main()
