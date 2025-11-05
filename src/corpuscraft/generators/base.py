"""Base generator class."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from corpuscraft.llm.base import BaseLLM
from corpuscraft.parsers.docling_parser import ParsedDocument

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Abstract base class for dataset generators."""

    def __init__(self, llm: BaseLLM, **kwargs: Any) -> None:
        """Initialize generator.

        Args:
            llm: LLM backend for generation
            **kwargs: Additional generator-specific parameters
        """
        self.llm = llm
        self.kwargs = kwargs
        logger.info(f"Initialized {self.__class__.__name__} with {llm}")

    @abstractmethod
    def generate(
        self,
        documents: list[ParsedDocument],
        num_examples: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate dataset examples from documents.

        Args:
            documents: List of parsed documents
            num_examples: Number of examples to generate
            **kwargs: Additional generation parameters

        Returns:
            List of generated examples (dicts)
        """
        pass

    @abstractmethod
    def get_example_schema(self) -> dict[str, type]:
        """Get the schema of generated examples.

        Returns:
            Dictionary mapping field names to types
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(llm={self.llm})"
