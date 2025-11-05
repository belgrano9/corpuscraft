"""Base LLM interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for LLM backends."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Initialize LLM backend.

        Args:
            model: Model name/identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional backend-specific parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text from multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(model={self.model})"
