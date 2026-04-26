"""
Test VLM Document Processing

This script demonstrates how to use the pre-configured Ollama VLM converter
from ollama_configs.py to process documents using Vision Language Models.

VLM (Vision Language Model) processing provides better results for complex
documents with formulas, tables, and mixed layouts compared to traditional OCR.

Prerequisites:
    - docling library installed
    - Ollama installed and running
    - Granite document model: `ollama pull gabegoodhart/granite-docling:258M`
    - Internet connection (for downloading PDF)

Usage:
    python test_vlm.py

Note:
    For VLM use, always prioritize Ollama or vLLM for local processing.
"""

import logging

from ollama_configs import check_ollama_connection, converter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_URL = "https://arxiv.org/pdf/2408.09869"  # Test document


def main():
    """Process a document using VLM and export to Markdown."""
    try:
        # Check Ollama connectivity before processing
        if not check_ollama_connection():
            logger.error("Cannot connect to Ollama. Please ensure it's running.")
            logger.info("Install: https://ollama.ai")
            logger.info("Pull model: ollama pull gabegoodhart/granite-docling:258M")
            return

        # Process document
        logger.info(f"Processing document: {SOURCE_URL}")
        logger.info("Using VLM pipeline with Ollama...")

        result = converter.convert(source=SOURCE_URL)
        doc = result.document

        # Export to Markdown
        markdown_output = doc.export_to_markdown()
        print("\n" + "=" * 80)
        print("DOCUMENT OUTPUT (Markdown)")
        print("=" * 80)
        print(markdown_output)

        logger.info("✓ Document processing completed successfully")

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Make sure Ollama is running and the model is available")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
