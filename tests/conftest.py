"""Pytest configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from corpuscraft.config import (
    CorpusCraftConfig,
    GeneratorConfig,
    GeneratorType,
    InputConfig,
    LLMConfig,
    OutputConfig,
    ProcessingConfig,
    QAGeneratorConfig,
)
from corpuscraft.parsers.docling_parser import ParsedDocument


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that enables computers to learn
    from data without being explicitly programmed. It uses algorithms to identify patterns
    in large datasets and make predictions or decisions based on those patterns.

    There are three main types of machine learning:
    1. Supervised learning - learning from labeled data
    2. Unsupervised learning - finding patterns in unlabeled data
    3. Reinforcement learning - learning through trial and error

    Deep learning is a subset of machine learning that uses neural networks with multiple
    layers to process complex patterns in data.
    """


@pytest.fixture
def sample_chunks() -> list[str]:
    """Sample text chunks for testing."""
    return [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "It uses algorithms to identify patterns in large datasets and make predictions.",
        "There are three main types: supervised, unsupervised, and reinforcement learning.",
        "Deep learning uses neural networks with multiple layers.",
    ]


@pytest.fixture
def sample_parsed_document(sample_text: str, sample_chunks: list[str], temp_dir: Path) -> ParsedDocument:
    """Create a sample parsed document."""
    test_file = temp_dir / "test_document.txt"
    test_file.write_text(sample_text)

    return ParsedDocument(
        text=sample_text,
        chunks=sample_chunks,
        metadata={
            "file_name": "test_document.txt",
            "file_type": "txt",
            "file_path": str(test_file),
        },
        file_path=test_file,
    )


@pytest.fixture
def sample_qa_pairs() -> list[dict[str, Any]]:
    """Sample QA pairs for testing."""
    return [
        {
            "question": "What is machine learning?",
            "answer": "A subset of AI that enables computers to learn from data.",
            "context": "Machine learning is a subset of artificial intelligence...",
            "difficulty": "easy",
            "type": "factual",
            "source_file": "test_document.txt",
            "source_metadata": {"file_name": "test_document.txt"},
        },
        {
            "question": "What are the three main types of machine learning?",
            "answer": "Supervised learning, unsupervised learning, and reinforcement learning.",
            "context": "There are three main types of machine learning...",
            "difficulty": "medium",
            "type": "factual",
            "source_file": "test_document.txt",
            "source_metadata": {"file_name": "test_document.txt"},
        },
    ]


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Mock Ollama client for testing."""
    client = MagicMock()

    # Mock list() response
    client.list.return_value = {
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "mistral"},
            {"name": "qwen2.5"},
        ]
    }

    # Mock generate() response
    client.generate.return_value = {
        "response": json.dumps([
            {
                "question": "What is machine learning?",
                "answer": "A subset of AI that enables computers to learn from data.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])
    }

    # Mock chat() response
    client.chat.return_value = {
        "message": {
            "content": "This is a chat response."
        }
    }

    return client


@pytest.fixture
def input_config(temp_dir: Path) -> InputConfig:
    """Create sample input configuration."""
    return InputConfig(
        folder=temp_dir,
        file_types=["pdf", "txt", "md"],
        recursive=True,
    )


@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Create sample processing configuration."""
    return ProcessingConfig(
        chunk_size=512,
        chunk_overlap=50,
        ocr_enabled=True,
        extract_tables=True,
        extract_images=False,
    )


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create sample LLM configuration."""
    return LLMConfig(
        backend="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def qa_generator_config() -> QAGeneratorConfig:
    """Create sample QA generator configuration."""
    return QAGeneratorConfig(
        num_examples=10,
        question_types=["factual", "reasoning"],
        difficulty_levels=["easy", "medium"],
        min_answer_length=5,
        max_answer_length=50,
    )


@pytest.fixture
def output_config(temp_dir: Path) -> OutputConfig:
    """Create sample output configuration."""
    return OutputConfig(
        format="jsonl",
        output_dir=temp_dir / "output",
        split_ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        seed=42,
    )


@pytest.fixture
def full_config(
    input_config: InputConfig,
    processing_config: ProcessingConfig,
    llm_config: LLMConfig,
    output_config: OutputConfig,
) -> CorpusCraftConfig:
    """Create complete CorpusCraft configuration."""
    return CorpusCraftConfig(
        input=input_config,
        processing=processing_config,
        llm=llm_config,
        generators=[
            GeneratorConfig(
                type=GeneratorType.QA,
                qa=QAGeneratorConfig(num_examples=10),
            )
        ],
        output=output_config,
    )


@pytest.fixture
def sample_jsonl_data() -> list[dict[str, Any]]:
    """Sample JSONL data for testing."""
    return [
        {"id": i, "text": f"Sample text {i}", "label": i % 3}
        for i in range(100)
    ]
