"""CorpusCraft: Transform your documents into training datasets."""

__version__ = "0.1.0"

from corpuscraft.generators import BaseGenerator, QAGenerator
from corpuscraft.llm import BaseLLM, OllamaLLM
from corpuscraft.output import JSONLWriter
from corpuscraft.parsers import DoclingParser

__all__ = [
    "BaseGenerator",
    "QAGenerator",
    "BaseLLM",
    "OllamaLLM",
    "JSONLWriter",
    "DoclingParser",
]
