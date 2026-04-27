from pathlib import Path

from corpuscraft.preprocessing.poppler import PopplerPreprocessor


def main() -> None:
    pdf_path = Path("data/raw/bulletin-de-paie-du-011025-au-311025.pdf")
    output_dir = Path("outputs/preprocessed") / pdf_path.stem

    # clean=True  → pdftocairo strips invisible text, annotations, comments
    # split=True  → pdfseparate produces one PDF per page
    # rasterize=True → pdftoppm renders pages to PNG at 150 DPI
    pre = PopplerPreprocessor(
        clean=True,
        split=True,
        rasterize=True,
        raster_dpi=150,
        raster_format="png",
    )

    print(f"Preprocessing: {pdf_path}")
    result = pre.run(pdf_path, output_dir)

    print(f"\n{result}")
    print(f"  Pages       : {result.metadata.page_count}")
    print(f"  Encrypted   : {result.metadata.encrypted}")
    print(f"  Page size   : {result.metadata.page_size}")
    print(f"  PDF version : {result.metadata.pdf_version}")
    print(f"  Title       : {result.metadata.title or '(none)'}")
    print(f"  Author      : {result.metadata.author or '(none)'}")
    print(f"  Scanned     : {result.is_scanned}")
    print(f"  Cleaned PDF : {result.cleaned_pdf}")
    print(f"  Page PDFs   : {len(result.page_pdfs)} file(s)")
    print(f"  Page images : {len(result.page_images)} file(s)")

    print(f"\nFeed to parser → {result.parser_input()}")

    if result.is_scanned:
        print("\nHint: no native text layer detected — use pipeline='ocr' or pipeline='vlm'")
    else:
        print("\nHint: native text detected — pipeline='standard', 'pymupdf', or 'pdfplumber' will work well")


if __name__ == "__main__":
    main()
