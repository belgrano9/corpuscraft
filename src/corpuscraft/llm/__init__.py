"""LLM backend integrations."""

from corpuscraft.llm.base import BaseLLM
from corpuscraft.llm.ollama import OllamaLLM

__all__ = ["BaseLLM", "OllamaLLM"]
