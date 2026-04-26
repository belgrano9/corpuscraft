"""
Multi-Language OCR Processing

This script demonstrates how to process scanned/image-based PDFs with OCR
using Tesseract with specific language support. It's configured for French
documents but can be adapted for any language supported by Tesseract.

This example uses forced full-page OCR, which is useful for:
- Scanned documents
- Image-based PDFs without embedded text
- Documents with poor text extraction quality

Prerequisites:
    - docling library installed
    - Tesseract OCR installed (https://github.com/tesseract-ocr/tesseract)
    - Tesseract language packs installed (e.g., `apt install tesseract-ocr-fra`)
    - French payslip PDF in data/raw/ directory

Usage:
    python translation_ocr.py

Language Configuration:
    - French: lang=["fra"]
    - English: lang=["eng"]
    - Auto-detect: lang=["auto"]
    - Multiple: lang=["fra", "eng"]
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

# Configuration
SOURCE_FILE = Path("data") / "raw" / "bulletin-de-paie-du-011025-au-311025.pdf"
DOCUMENT_NAME = "bulletin-de-paie-du-011025-au-311025"
OCR_LANGUAGE = ["fra"]  # French language pack
OUTPUT_DIR = Path("output")


def main():
    """
    Process a French PDF document with Tesseract OCR and export to HTML.

    The script uses forced full-page OCR which is ideal for scanned documents
    or image-based PDFs without embedded text layers.
    """
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Processing document: {SOURCE_FILE}")
        print(f"OCR Language: {', '.join(OCR_LANGUAGE)}")
        print(f"Using Tesseract CLI with forced full-page OCR")

        # Configure Tesseract OCR with French language support
        ocr_options = TesseractCliOcrOptions(lang=OCR_LANGUAGE)

        # Configure pipeline for forced OCR
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            force_full_page_ocr=True,  # Force OCR on all pages, even if text exists
            ocr_options=ocr_options
        )

        # Create converter with OCR-enabled pipeline
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        # Convert document
        print("Converting document with OCR...")
        result = converter.convert(str(SOURCE_FILE))
        doc = result.document

        # Save as HTML with referenced images
        html_filename = OUTPUT_DIR / f"{DOCUMENT_NAME}-with-image-refs.html"
        doc.save_as_html(str(html_filename), image_mode=ImageRefMode.REFERENCED)

        print(f"✓ Document saved to: {html_filename.absolute()}")
        print(f"  HTML file includes referenced images")

    except FileNotFoundError:
        print(f"Error: Source file not found: {SOURCE_FILE}")
        print("Make sure the data/raw/ directory exists with the PDF file")
    except RuntimeError as e:
        if "tesseract" in str(e).lower():
            print(f"Tesseract Error: {e}")
            print("Make sure Tesseract is installed:")
            print("  - Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-fra")
            print("  - macOS: brew install tesseract tesseract-lang")
            print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        else:
            raise
    except Exception as e:
        print(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
