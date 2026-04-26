"""
Ollama VLM Configuration Module

This module provides a pre-configured DocumentConverter for use with Ollama's Vision Language Models.
It's designed as a reusable configuration that can be imported by other scripts.

The configuration uses:
- Granite Document OCR model via Ollama
- DOCTAGS response format for better document structure
- Higher resolution scaling for improved accuracy
- Support for both PDF and image inputs

Prerequisites:
    - docling library installed
    - Ollama installed and running (https://ollama.ai)
    - Granite document model pulled: `ollama pull gabegoodhart/granite-docling:258M`

Usage:
    from ollama_configs import converter

    result = converter.convert("document.pdf")
    doc = result.document
    print(doc.export_to_markdown())

Environment Variables:
    OLLAMA_HOST: Override default Ollama endpoint (default: http://localhost:11434)
    OLLAMA_MODEL: Override default model (default: gabegoodhart/granite-docling:258M)
"""

import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Configuration constants
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_ENDPOINT = f"{OLLAMA_HOST}/v1/chat/completions"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gabegoodhart/granite-docling:258M")
TIMEOUT_SECONDS = 300
SCALE_FACTOR = 2.0  # Higher resolution for better formula/table recognition

# VLM prompt for document extraction
DOCUMENT_EXTRACTION_PROMPT = (
    "Convert this page to docling format. "
    "Extract all text, preserving the exact layout and structure. "
    "Be precise and do not skip any content."
)

# Configure the VLM pipeline with Ollama support
pipeline_options = VlmPipelineOptions(
    enable_remote_services=True,  # Required when calling remote VLM endpoints
)

# Configure Ollama API options
pipeline_options.vlm_options = ApiVlmOptions(
    url=OLLAMA_ENDPOINT,
    params=dict(
        model=OLLAMA_MODEL,
    ),
    prompt=DOCUMENT_EXTRACTION_PROMPT,
    timeout=TIMEOUT_SECONDS,
    scale=SCALE_FACTOR,
    response_format=ResponseFormat.DOCTAGS,  # Use DOCTAGS format for better structure
)

# Create the DocumentConverter with Ollama configuration
# Supports both PDF and image formats
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        ),
        InputFormat.IMAGE: PdfFormatOption(  # Use same VLM pipeline for images
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        ),
    }
)


def check_ollama_connection():
    """
    Check if Ollama is accessible at the configured endpoint.

    Returns:
        bool: True if Ollama is reachable, False otherwise
    """
    import urllib.request
    try:
        urllib.request.urlopen(OLLAMA_HOST, timeout=5)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # When run directly, perform a connection check
    print(f"Ollama Configuration")
    print(f"  Endpoint: {OLLAMA_ENDPOINT}")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Timeout: {TIMEOUT_SECONDS}s")
    print(f"  Scale: {SCALE_FACTOR}x")
    print()

    if check_ollama_connection():
        print("✓ Ollama is accessible")
    else:
        print(f"✗ Cannot connect to Ollama at {OLLAMA_HOST}")
        print("  Make sure Ollama is running: https://ollama.ai")
