from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PdfPlumberParser(BaseParser):
    """
    A parser that uses pdfplumber to extract text and tables from PDFs.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        logger.info(f"Parsing {path.name} with pdfplumber")
        
        content_parts = []
        metadata = {}
        
        try:
            with pdfplumber.open(str(path)) as pdf:
                metadata.update(pdf.metadata)
                metadata["page_count"] = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    # Extract raw text
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
                    
                    # Optionally, you can extract tables. pdfplumber is great at this!
                    tables = page.extract_tables()
                    for table in tables:
                        # Formatting tables as basic markdown
                        if not table:
                            continue
                        
                        # Add a newline separator
                        content_parts.append("\n")
                        
                        for row_idx, row in enumerate(table):
                            # Replace None with empty string and clean up newlines inside cells
                            clean_row = [str(cell).replace("\n", " ") if cell is not None else "" for cell in row]
                            row_str = "| " + " | ".join(clean_row) + " |"
                            content_parts.append(row_str)
                            
                            # Add markdown table separator after header
                            if row_idx == 0:
                                separator = "| " + " | ".join(["---"] * len(clean_row)) + " |"
                                content_parts.append(separator)
                        
                        content_parts.append("\n")
            
            md_text = "\n".join(content_parts)
            
            return ParsedDocument(
                content=md_text,
                source_path=path,
                pipeline="pdfplumber",
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"pdfplumber failed on {path.name}: {e}")
            raise
