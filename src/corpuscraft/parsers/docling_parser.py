"""
Docling-based document parser for CorpusCraft.

Uses IBM's Docling library for enterprise-grade document parsing with support
for PDF, DOCX, PPTX, HTML, Markdown, and TXT files.

Optional Ollama VLM backend support for enhanced OCR with vision models.
"""

from pathlib import Path
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from loguru import logger

from corpuscraft.models import ParsedDocument


class DoclingParser:
    """
    Generic document parser using IBM's Docling library.

    Supports multiple document formats with advanced features like OCR,
    table extraction, and layout detection.

    Optional Ollama VLM backend for enhanced OCR using vision models.

    Attributes:
        converter: The Docling DocumentConverter instance
        use_ollama: Whether Ollama VLM backend is enabled
    """

    # Supported file extensions
    SUPPORTED_FORMATS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".bmp",
    }

    def __init__(
        self,
        use_ollama: bool = True,
        ollama_url: str = "http://localhost:11434/v1/chat/completions",
        ollama_model: str = "gabegoodhart/granite-docling:258M",
        ocr_engine: str = "tesseract",
        vlm_timeout: int = 300,
        vlm_scale: float = 2.0,
        custom_prompt: Optional[str] = None,
    ):
        """
        Initialize the Docling parser.

        Args:
            use_ollama: Enable Ollama VLM backend for enhanced OCR (default: True)
            ollama_url: Ollama API endpoint (default: localhost:11434)
            ollama_model: Ollama model to use for VLM (default: granite-docling:258M)
            ocr_engine: OCR engine to use - "tesseract" or "easyocr" (default: tesseract)
            vlm_timeout: Timeout in seconds for VLM requests (default: 300)
            vlm_scale: Resolution scale for better OCR (default: 2.0)
            custom_prompt: Custom prompt for VLM (default: optimized for docling format)
        """
        self.use_ollama = use_ollama

        if use_ollama:
            logger.info(f"Initializing DoclingParser with Ollama VLM backend")
            logger.info(f"  Model: {ollama_model}")
            logger.info(f"  OCR Engine: {ocr_engine}")
            logger.info(f"  URL: {ollama_url}")

            # Configure VLM pipeline options
            pipeline_options = VlmPipelineOptions(
                enable_remote_services=True,  # Required for remote VLM endpoints
                ocr_engine=ocr_engine,
            )

            # Default prompt optimized for document extraction
            if custom_prompt is None:
                custom_prompt = (
                    "Convert this page to docling format. "
                    "Extract all text, preserving the exact layout and structure. "
                    "For mathematical formulas and equations: "
                    "- Use proper LaTeX notation "
                    "- Preserve all mathematical symbols, Greek letters, subscripts, and superscripts "
                    "- Maintain equation numbering and references "
                    "- Include both inline formulas (using $) and display equations (using $$) "
                    "Be precise and do not skip any mathematical content."
                )

            # Configure Ollama API options
            pipeline_options.vlm_options = ApiVlmOptions(
                url=ollama_url,
                params=dict(model=ollama_model),
                prompt=custom_prompt,
                timeout=vlm_timeout,
                scale=vlm_scale,
                response_format=ResponseFormat.DOCTAGS,
            )

            # Create converter with VLM pipeline for PDFs and images
            self.converter = DocumentConverter(
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
        else:
            # Default converter without VLM
            self.converter = DocumentConverter()
            logger.info("DoclingParser initialized with default settings")

    def parse_file(self, file_path: str | Path) -> ParsedDocument | None:
        """
        Parse a single document file.

        Args:
            file_path: Path to the document file

        Returns:
            ParsedDocument object containing the parsed content and metadata,
            or None if parsing failed

        Raises:
            FileNotFoundError: If the file does not exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self._is_supported_format(file_path):
            logger.warning(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
            return None

        try:
            logger.info(f"Parsing file: {file_path}")

            # Convert the document using Docling
            result = self.converter.convert(str(file_path))

            # Extract the markdown text content
            content = result.document.export_to_markdown()

            # Build metadata
            metadata = {
                "file_path": str(file_path.absolute()),
                "file_name": file_path.name,
                "format": file_path.suffix.lstrip("."),
                "file_size": file_path.stat().st_size,
            }

            # Add page count if available
            if hasattr(result.document, "page_count"):
                metadata["page_count"] = result.document.page_count

            logger.info(
                f"Successfully parsed {file_path.name}: "
                f"{len(content)} characters"
            )

            return ParsedDocument(content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def parse_folder(
        self,
        folder_path: str | Path,
        recursive: bool = True,
    ) -> List[ParsedDocument]:
        """
        Parse all supported documents in a folder.

        Args:
            folder_path: Path to the folder containing documents
            recursive: If True, search subdirectories recursively

        Returns:
            List of ParsedDocument objects (excludes failed parses)

        Raises:
            NotADirectoryError: If the path is not a directory
        """
        folder_path = Path(folder_path)

        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        logger.info(f"Parsing folder: {folder_path} (recursive={recursive})")

        # Find all supported files
        files = []
        if recursive:
            for ext in self.SUPPORTED_FORMATS:
                files.extend(folder_path.rglob(f"*{ext}"))
        else:
            for ext in self.SUPPORTED_FORMATS:
                files.extend(folder_path.glob(f"*{ext}"))

        logger.info(f"Found {len(files)} supported files")

        # Parse each file
        parsed_docs = []
        for file_path in files:
            doc = self.parse_file(file_path)
            if doc is not None:
                parsed_docs.append(doc)

        logger.info(
            f"Successfully parsed {len(parsed_docs)}/{len(files)} documents"
        )

        return parsed_docs

    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if the file format is supported."""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS

    def get_supported_formats(self) -> List[str]:
        """Get a list of supported file formats."""
        return sorted(self.SUPPORTED_FORMATS)
