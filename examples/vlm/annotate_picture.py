"""
Picture Annotation with VLM

This script demonstrates how to extract and annotate picture elements from documents
using Ollama's Vision Language Models. It processes documents and extracts metadata
about images, figures, and diagrams, including captions and annotations.

The script uses the IBM Granite Vision model which is optimized for understanding
visual content in documents.

Prerequisites:
    - docling library installed
    - Ollama installed and running
    - Granite Vision model: `ollama pull ibm/granite3.3-vision:2b`
    - Internet connection (for downloading PDF)

Usage:
    python annotate_picture.py

Environment Variables:
    OLLAMA_HOST: Override default Ollama endpoint (default: http://localhost:11434)
    VISION_MODEL: Override default vision model (default: ibm/granite3.3-vision:2b)
"""

import logging
import os

from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_ENDPOINT = f"{OLLAMA_HOST}/v1/chat/completions"
DEFAULT_MODEL = "ibm/granite3.3-vision:2b"
VISION_MODEL = os.getenv("VISION_MODEL", DEFAULT_MODEL)
SOURCE_URL = "https://arxiv.org/pdf/2408.09869"


def create_ollama_vlm_options(model: str = None):
    """
    Create VLM pipeline options configured for Ollama with vision capabilities.

    This configuration is optimized for extracting visual content from documents,
    including images, diagrams, charts, and their associated metadata.

    Args:
        model: The Ollama vision model to use. If None, uses VISION_MODEL env var
               or default: ibm/granite3.3-vision:2b

    Returns:
        VlmPipelineOptions configured for Ollama with vision model
    """
    if model is None:
        model = VISION_MODEL

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True,  # Required when calling remote VLM endpoints
    )

    # Configure Ollama API options for vision model
    pipeline_options.vlm_options = ApiVlmOptions(
        url=OLLAMA_ENDPOINT,
        params=dict(
            model=model,
            seed=42,
        ),
        prompt=(
            "Convert this page to docling format. "
            "Extract all text, preserving the exact layout and structure. "
            "Be precise and do not skip any content."
        ),
        timeout=300,
        scale=2.0,  # Higher resolution for better image recognition
        response_format=ResponseFormat.DOCTAGS,
    )

    return pipeline_options


def main():
    """
    Process a document and extract picture annotations using VLM.

    The function converts a document and iterates through all elements,
    printing detailed information about each picture/figure found.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Configuring VLM with model: {VISION_MODEL}")
        logger.info(f"Processing document: {SOURCE_URL}")

        # Create VLM pipeline options
        pipeline_options = create_ollama_vlm_options()

        # Create DocumentConverter with VLM pipeline
        # Supports both PDF and image formats
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                ),
            }
        )

        # Convert document
        logger.info("Converting document with VLM pipeline...")
        result = doc_converter.convert(SOURCE_URL)
        doc = result.document

        # Extract and display picture annotations
        picture_count = 0
        print("\n" + "=" * 80)
        print("EXTRACTED PICTURE ANNOTATIONS")
        print("=" * 80 + "\n")

        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                picture_count += 1
                print(f"Picture #{picture_count}: {element.self_ref}")
                print(f"  Caption: {element.caption_text(doc=doc)}")
                print(f"  Annotations: {element.annotations}")
                print("-" * 80)

        if picture_count == 0:
            logger.info("No pictures found in document")
        else:
            logger.info(f"✓ Successfully extracted {picture_count} picture(s)")

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error(f"Make sure Ollama is running at {OLLAMA_HOST}")
        logger.error(f"Pull model: ollama pull {VISION_MODEL}")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


if __name__ == "__main__":
    main()
