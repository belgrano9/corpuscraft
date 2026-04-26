"""
OCR Engine Comparison

This script demonstrates how to force full-page OCR with different OCR engines
supported by Docling. It's useful for testing which OCR engine works best for
your specific documents.

The script also enables table structure detection with cell matching, which is
useful for extracting structured data from scanned tables.

Supported OCR Engines:
    - RapidOCR: Fast, lightweight, pure Python (default in this example)
    - TesseractCLI: Industry standard, requires Tesseract installation
    - Tesseract: Python bindings to Tesseract
    - EasyOCR: Deep learning-based, good for complex text
    - OcrMac: macOS native OCR (macOS only)

Prerequisites:
    - docling library installed
    - Specific OCR engine dependencies (see below)
    - French payslip PDF in data/raw/ directory

Usage:
    python force_ocr.py

Engine-Specific Requirements:
    - RapidOCR: pip install rapidocr-onnxruntime
    - TesseractCLI: Tesseract installed on system
    - EasyOCR: pip install easyocr
    - OcrMac: macOS 10.15+ (built-in)
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    # Uncomment the OCR engine you want to test:
    # TesseractCliOcrOptions,
    # TesseractOcrOptions,
    # EasyOcrOptions,
    # OcrMacOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configuration
SOURCE_FILE = Path("data") / "raw" / "bulletin-de-paie-du-011025-au-311025.pdf"


def main():
    """
    Process a document with forced OCR and table structure detection.

    This script tests OCR engines on a scanned document. Uncomment different
    OCR options below to compare results.
    """
    try:
        print(f"Processing document: {SOURCE_FILE}")
        print("Using: RapidOCR with forced full-page OCR")
        print("Features: Table structure detection with cell matching")

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        # ===== OCR Engine Selection =====
        # Uncomment ONE of the following options to test different OCR engines:

        # Option 1: RapidOCR - Fast, lightweight, pure Python (CURRENT)
        ocr_options = RapidOcrOptions(force_full_page_ocr=True)

        # Option 2: Tesseract CLI - Industry standard, requires installation
        # ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)

        # Option 3: Tesseract Python bindings
        # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)

        # Option 4: EasyOCR - Deep learning-based, good for complex layouts
        # ocr_options = EasyOcrOptions(force_full_page_ocr=True)

        # Option 5: macOS native OCR (macOS 10.15+ only)
        # ocr_options = OcrMacOptions(force_full_page_ocr=True)

        pipeline_options.ocr_options = ocr_options

        # Create converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

        # Convert document
        print("\nConverting document with OCR...")
        result = converter.convert(str(SOURCE_FILE))
        doc = result.document

        # Export to Markdown
        markdown_output = doc.export_to_markdown()

        print("\n" + "=" * 80)
        print("DOCUMENT OUTPUT (Markdown)")
        print("=" * 80)
        print(markdown_output)
        print("\n✓ OCR processing completed successfully")

    except FileNotFoundError:
        print(f"Error: Source file not found: {SOURCE_FILE}")
        print("Make sure the data/raw/ directory exists with the PDF file")
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nMake sure the required OCR engine is installed:")
        print("  - RapidOCR: pip install rapidocr-onnxruntime")
        print("  - EasyOCR: pip install easyocr")
        print("  - Tesseract: Install from https://github.com/tesseract-ocr/tesseract")
    except Exception as e:
        print(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
