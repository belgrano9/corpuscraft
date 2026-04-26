from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    content: str
    source_path: Path
    pipeline: str
    metadata: dict = field(default_factory=dict)
    chunks: list[str] | None = None

    def __len__(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return (
            f"ParsedDocument(source={self.source_path.name!r}, "
            f"pipeline={self.pipeline!r}, chars={len(self)})"
        )


@dataclass
class QAExample:
    question: str
    answer: str
    context: str
    source: str
    difficulty: str = "medium"
