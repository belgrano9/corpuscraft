"""
Simple Document Processor

This is the most basic example of using Docling to convert a PDF document to Markdown.
It demonstrates the minimal code needed to get started with document processing.

Prerequisites:
    - docling library installed
    - Internet connection (for downloading PDF)

Usage:
    python simple_processor.py
"""

from docling.document_converter import DocumentConverter

# Configuration
SOURCE_URL = "https://arxiv.org/pdf/2408.09869"  # file path or URL


def main():
    """Convert a PDF document to Markdown using default Docling settings."""
    try:
        # Initialize converter with default settings
        converter = DocumentConverter()

        # Convert the document
        print(f"Converting document from: {SOURCE_URL}")
        result = converter.convert(SOURCE_URL)
        doc = result.document

        # Export to Markdown and print
        markdown_output = doc.export_to_markdown()
        print(markdown_output)

    except Exception as e:
        print(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
