from __future__ import annotations

import logging
from pathlib import Path

import pymupdf4llm
import fitz

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """
    A pure PyMuPDF parser that leverages pymupdf4llm for markdown conversion.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        logger.info(f"Parsing {path.name} with PyMuPDF")
        
        # We can use pymupdf4llm to easily extract high-quality markdown
        try:
            try:
                md_text = pymupdf4llm.to_markdown(str(path))
            except AttributeError as e:
                # Fallback if RapidOCR integration fails due to dependency conflicts (e.g. text_detector)
                logger.warning(f"PyMuPDF OCR failed, falling back to use_ocr=0: {e}")
                md_text = pymupdf4llm.to_markdown(str(path), use_ocr=0)
            
            # Optional: Extract basic metadata using pure fitz (PyMuPDF)
            metadata = {}
            with fitz.open(str(path)) as doc:
                metadata.update(doc.metadata)
                metadata["page_count"] = doc.page_count
            
            return ParsedDocument(
                content=md_text,
                source_path=path,
                pipeline="pymupdf",
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"PyMuPDF failed on {path.name}: {e}")
            raise
