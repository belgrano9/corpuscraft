from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from corpuscraft.models import ParsedDocument

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".html", ".htm",
    ".md", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, path: Path) -> ParsedDocument: ...

    def parse_folder(
        self, folder: Path, glob: str = "**/*"
    ) -> list[ParsedDocument]:
        paths = [
            p for p in folder.glob(glob)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return [self.parse_file(p) for p in sorted(paths)]
