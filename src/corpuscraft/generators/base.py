from __future__ import annotations

from abc import ABC, abstractmethod

from corpuscraft.config import GeneratorConfig, LLMConfig
from corpuscraft.models import ParsedDocument, QAExample


class BaseGenerator(ABC):
    def __init__(self, config: GeneratorConfig, llm: LLMConfig) -> None:
        self.config = config
        self.llm = llm

    @abstractmethod
    def generate(self, document: ParsedDocument) -> list[QAExample]: ...
