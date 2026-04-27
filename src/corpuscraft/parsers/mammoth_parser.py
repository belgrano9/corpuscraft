from __future__ import annotations

import logging
from pathlib import Path

import mammoth

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class MammothParser(BaseParser):
    """DOCX → markdown via mammoth.

    Mammoth focuses on semantic structure (headings, lists, tables, links)
    rather than visual fidelity. Output is clean markdown suitable for LLM
    training data — formatting noise like fonts and colors is dropped.
    DOCX only; legacy .doc binary format is not supported.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() != ".docx":
            raise ValueError(
                f"MammothParser only handles .docx files, got {path.suffix}"
            )

        logger.info(f"Parsing {path.name} with mammoth")

        with open(path, "rb") as f:
            result = mammoth.convert_to_markdown(f)

        warnings = [m.message for m in result.messages if m.type == "warning"]
        if warnings:
            logger.debug(f"mammoth produced {len(warnings)} warnings on {path.name}")

        metadata = {
            "file_size": path.stat().st_size,
            "warnings": warnings,
            "warning_count": len(warnings),
        }

        return ParsedDocument(
            content=result.value,
            source_path=path,
            pipeline="mammoth",
            metadata=metadata,
        )
