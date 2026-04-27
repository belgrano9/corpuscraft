from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PDFMetadata:
    page_count: int
    encrypted: bool
    pdf_version: str
    title: str
    author: str
    page_size: str
    raw: dict = field(default_factory=dict)


@dataclass
class PreprocessedPDF:
    source_path: Path
    output_dir: Path
    metadata: PDFMetadata
    is_scanned: bool
    cleaned_pdf: Path | None = None
    page_pdfs: list[Path] = field(default_factory=list)
    page_images: list[Path] = field(default_factory=list)

    def parser_input(self) -> Path:
        """Best path to feed into a parser: cleaned PDF if available, else original."""
        return self.cleaned_pdf if self.cleaned_pdf is not None else self.source_path

    def __repr__(self) -> str:
        return (
            f"PreprocessedPDF(source={self.source_path.name!r}, "
            f"pages={self.metadata.page_count}, "
            f"scanned={self.is_scanned}, "
            f"cleaned={self.cleaned_pdf is not None})"
        )
