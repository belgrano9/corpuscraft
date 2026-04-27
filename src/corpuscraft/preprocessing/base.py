from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from corpuscraft.preprocessing.models import PreprocessedPDF


class BasePreprocessor(ABC):
    @abstractmethod
    def run(self, pdf_path: Path, output_dir: Path) -> PreprocessedPDF: ...

    def run_folder(
        self, folder: Path, output_dir: Path
    ) -> list[PreprocessedPDF]:
        pdfs = sorted(folder.glob("**/*.pdf"))
        return [self.run(p, output_dir / p.stem) for p in pdfs]
