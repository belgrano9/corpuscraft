from __future__ import annotations

import logging
from pathlib import Path

from markitdown import MarkItDown

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class MarkItDownParser(BaseParser):
    """Unified Office → markdown via Microsoft's markitdown.

    Handles DOCX, PPTX, XLSX, HTML and several other formats with one API.
    Output is markdown with headings, lists, and tables preserved. Use this
    when you want a single parser covering multiple Office formats and don't
    need the structural fidelity of python-docx / python-pptx.
    """

    def __init__(self, config: ParserConfig):
        self.config = config
        self._converter = MarkItDown()

    def parse_file(self, path: Path) -> ParsedDocument:
        logger.info(f"Parsing {path.name} with markitdown")

        result = self._converter.convert(str(path))

        metadata = {
            "file_size": path.stat().st_size,
            "extension": path.suffix.lower(),
        }

        return ParsedDocument(
            content=result.text_content,
            source_path=path,
            pipeline="markitdown",
            metadata=metadata,
        )
