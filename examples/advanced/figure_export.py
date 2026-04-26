"""
Comprehensive Figure and Table Export

This script demonstrates a complete document processing pipeline that extracts
and exports all visual elements from a PDF document, including:
- Individual page images
- Table images
- Figure/picture images
- Multiple output formats (Markdown with embedded/referenced images, HTML)

This is useful for:
- Creating datasets from documents
- Extracting visual content for analysis
- Generating multiple output formats from a single source

Prerequisites:
    - docling library installed
    - PIL/Pillow for image handling
    - Source PDF document

Usage:
    python figure_export.py

Output:
    All files are saved to the scratch/ directory:
    - {filename}-{page_no}.png: Individual page images
    - {filename}-table-{n}.png: Table images
    - {filename}-picture-{n}.png: Figure images
    - {filename}-with-images.md: Markdown with embedded images
    - {filename}-with-image-refs.md: Markdown with image references
    - {filename}-with-image-refs.html: HTML with image references
"""

import logging
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configuration
SOURCE_FILE = Path("data") / "raw" / "bulletin-de-paie-du-011025-au-311025.pdf"
# Alternative: Use arXiv URL
# SOURCE_FILE = "https://arxiv.org/pdf/2408.09869"

OUTPUT_DIR = Path("scratch")
IMAGE_RESOLUTION_SCALE = 2.0  # Scale factor for image resolution (1.0 = 72 DPI)

# Setup logging
_log = logging.getLogger(__name__)


def main():
    """
    Process a document and export all figures, tables, and pages as images.

    This function demonstrates a complete export pipeline, generating multiple
    output formats and extracting all visual elements from the document.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        _log.info(f"Processing document: {SOURCE_FILE}")
        _log.info(f"Output directory: {OUTPUT_DIR}")
        _log.info(f"Image resolution scale: {IMAGE_RESOLUTION_SCALE}x")

        # Configure pipeline to generate images for pages and pictures
        # The `images_scale` controls rendered image resolution (scale=1 ~ 72 DPI)
        # The `generate_*` toggles decide which elements are enriched with images
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True

        # Create document converter
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Convert document
        start_time = time.time()
        _log.info("Converting document...")
        conv_res = doc_converter.convert(str(SOURCE_FILE))

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        doc_filename = conv_res.input.file.stem

        # Export page images
        _log.info(f"Exporting {len(conv_res.document.pages)} page images...")
        for page_no, page in conv_res.document.pages.items():
            page_no = page.page_no
            page_image_filename = OUTPUT_DIR / f"{doc_filename}-page-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

        # Export table and picture images
        _log.info("Extracting tables and figures...")
        table_counter = 0
        picture_counter = 0

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = OUTPUT_DIR / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

            if isinstance(element, PictureItem):
                picture_counter += 1
                element_image_filename = OUTPUT_DIR / f"{doc_filename}-picture-{picture_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conv_res.document).save(fp, "PNG")

        _log.info(f"  Extracted {table_counter} table(s)")
        _log.info(f"  Extracted {picture_counter} picture(s)")

        # Export in multiple formats
        _log.info("Exporting document in multiple formats...")

        # 1. Markdown with embedded images (base64-encoded)
        md_embedded_filename = OUTPUT_DIR / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_embedded_filename, image_mode=ImageRefMode.EMBEDDED)
        _log.info(f"  ✓ Markdown (embedded images): {md_embedded_filename}")

        # 2. Markdown with externally referenced images
        md_refs_filename = OUTPUT_DIR / f"{doc_filename}-with-image-refs.md"
        conv_res.document.save_as_markdown(md_refs_filename, image_mode=ImageRefMode.REFERENCED)
        _log.info(f"  ✓ Markdown (referenced images): {md_refs_filename}")

        # 3. HTML with externally referenced images
        html_filename = OUTPUT_DIR / f"{doc_filename}-with-image-refs.html"
        conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)
        _log.info(f"  ✓ HTML (referenced images): {html_filename}")

        # Performance summary
        end_time = time.time() - start_time
        _log.info(f"\n{'='*60}")
        _log.info(f"✓ Document processing completed in {end_time:.2f} seconds")
        _log.info(f"{'='*60}")
        _log.info(f"Summary:")
        _log.info(f"  - Pages: {len(conv_res.document.pages)}")
        _log.info(f"  - Tables: {table_counter}")
        _log.info(f"  - Pictures: {picture_counter}")
        _log.info(f"  - Output directory: {OUTPUT_DIR.absolute()}")

    except FileNotFoundError:
        _log.error(f"Error: Source file not found: {SOURCE_FILE}")
        _log.error("Make sure the data/raw/ directory exists with the PDF file")
    except Exception as e:
        _log.error(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
