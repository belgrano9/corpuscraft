"""
GPU-Accelerated Document Processor

This example demonstrates how to use GPU acceleration for faster document processing.
It uses CUDA-enabled device acceleration and batch processing for optimal performance.

Prerequisites:
    - docling library installed
    - NVIDIA GPU with CUDA support
    - CUDA toolkit installed
    - Internet connection (for downloading PDF)

Usage:
    python simple_processor_gpu.py

Note:
    OCR is disabled in this example. If you need OCR, set `pipeline_options.do_ocr = True`
    and the configured batch sizes will be used for OCR processing.
"""

from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

# Configuration
SOURCE_URL = "https://arxiv.org/pdf/2408.09869"  # 8-page PDF
OUTPUT_FILE = "output.md"

# Batch sizes for GPU processing (used when OCR is enabled)
OCR_BATCH_SIZE = 4
LAYOUT_BATCH_SIZE = 64
TABLE_BATCH_SIZE = 4


def main():
    """
    Convert a PDF document to Markdown using GPU acceleration.

    The pipeline uses CUDA for accelerated processing of layout detection,
    table extraction, and other computationally intensive tasks.
    """
    try:
        # Configure GPU-accelerated pipeline
        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
            ),
            ocr_batch_size=OCR_BATCH_SIZE,
            layout_batch_size=LAYOUT_BATCH_SIZE,
            table_batch_size=TABLE_BATCH_SIZE,
        )

        # Disable OCR for this example (remove if you need OCR)
        pipeline_options.do_ocr = False

        # Create converter with GPU-enabled pipeline
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        # Convert document
        print(f"Converting document from: {SOURCE_URL}")
        print(f"Using GPU acceleration: CUDA")
        result = doc_converter.convert(SOURCE_URL)
        doc = result.document

        # Save output to file
        output_path = Path(OUTPUT_FILE)
        doc.save_as_markdown(str(output_path))
        print(f"✓ Document saved to: {output_path.absolute()}")

    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"GPU Error: {e}")
            print("Make sure you have a CUDA-capable GPU and CUDA toolkit installed.")
        else:
            raise
    except Exception as e:
        print(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
