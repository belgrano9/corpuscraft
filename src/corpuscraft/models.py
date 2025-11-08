"""
Data models for CorpusCraft.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with its content and metadata.

    Attributes:
        content: The full extracted text content from the document
        metadata: Dictionary containing document metadata (file path, format, etc.)
        chunks: Optional list of text chunks if the document was split
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)

    @property
    def file_path(self) -> Path | None:
        """Get the source file path from metadata."""
        path = self.metadata.get("file_path")
        return Path(path) if path else None

    @property
    def file_format(self) -> str | None:
        """Get the document format from metadata."""
        return self.metadata.get("format")

    @property
    def page_count(self) -> int | None:
        """Get the page count from metadata."""
        return self.metadata.get("page_count")

    def __len__(self) -> int:
        """Return the character count of the content."""
        return len(self.content)

    def __repr__(self) -> str:
        """Return a readable representation."""
        file_name = self.file_path.name if self.file_path else "Unknown"
        char_count = len(self.content)
        chunk_count = len(self.chunks)
        return (
            f"ParsedDocument(file='{file_name}', "
            f"format='{self.file_format}', "
            f"chars={char_count}, "
            f"chunks={chunk_count})"
        )
