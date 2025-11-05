"""Ollama LLM backend integration."""

import logging
from typing import Any

from ollama import Client

from corpuscraft.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM backend."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        """Initialize Ollama backend.

        Args:
            model: Model name (e.g., 'llama3.1:8b', 'mistral', 'qwen2.5')
            base_url: Ollama server URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.base_url = base_url
        self.client = Client(host=base_url)

        # Verify model is available
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify that the model is available locally."""
        try:
            models = self.client.list()
            available_models = [m["name"] for m in models.get("models", [])]

            if self.model not in available_models:
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Available models: {available_models}. "
                    f"You may need to run: ollama pull {self.model}"
                )
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            logger.info("Make sure Ollama is running: ollama serve")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            )
            return response["response"]
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    def batch_generate(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Generate text from multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                results.append("")

        return results

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                },
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
