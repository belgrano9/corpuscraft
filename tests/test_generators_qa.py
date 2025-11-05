"""Tests for QA generator."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corpuscraft.generators.qa_generator import QAGenerator
from corpuscraft.parsers.docling_parser import ParsedDocument


class TestQAGenerator:
    """Tests for QAGenerator class."""

    def test_initialization(self) -> None:
        """Test QAGenerator initialization."""
        mock_llm = MagicMock()

        generator = QAGenerator(
            llm=mock_llm,
            question_types=["factual", "reasoning"],
            difficulty_levels=["easy", "medium"],
            min_answer_length=5,
            max_answer_length=50,
        )

        assert generator.llm == mock_llm
        assert generator.question_types == ["factual", "reasoning"]
        assert generator.difficulty_levels == ["easy", "medium"]
        assert generator.min_answer_length == 5
        assert generator.max_answer_length == 50

    def test_initialization_with_defaults(self) -> None:
        """Test QAGenerator initialization with defaults."""
        mock_llm = MagicMock()
        generator = QAGenerator(llm=mock_llm)

        assert generator.question_types == ["factual", "reasoning", "comparison"]
        assert generator.difficulty_levels == ["easy", "medium", "hard"]
        assert generator.min_answer_length == 1
        assert generator.max_answer_length == 100

    def test_get_example_schema(self) -> None:
        """Test example schema."""
        mock_llm = MagicMock()
        generator = QAGenerator(llm=mock_llm)
        schema = generator.get_example_schema()

        assert schema["question"] == str
        assert schema["answer"] == str
        assert schema["context"] == str
        assert schema["difficulty"] == str
        assert schema["type"] == str
        assert schema["source_file"] == str
        assert schema["source_metadata"] == dict

    def test_generate_empty_documents(self) -> None:
        """Test generation with empty document list."""
        mock_llm = MagicMock()
        generator = QAGenerator(llm=mock_llm)

        result = generator.generate(documents=[], num_examples=10)

        assert result == []

    def test_generate_documents_without_chunks(self, tmp_path: Path) -> None:
        """Test generation with documents that have no chunks."""
        mock_llm = MagicMock()
        generator = QAGenerator(llm=mock_llm)

        doc = ParsedDocument(
            text="Some text",
            chunks=[],  # No chunks
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        result = generator.generate(documents=[doc], num_examples=10)

        assert result == []

    def test_generate_success(self, tmp_path: Path) -> None:
        """Test successful QA generation."""
        # Mock LLM
        mock_llm = MagicMock()
        qa_response = json.dumps([
            {
                "question": "What is machine learning?",
                "answer": "A subset of AI.",
                "difficulty": "easy",
                "type": "factual",
            },
            {
                "question": "What are the types?",
                "answer": "Supervised, unsupervised, and reinforcement.",
                "difficulty": "medium",
                "type": "factual",
            },
        ])
        mock_llm.generate.return_value = qa_response

        generator = QAGenerator(llm=mock_llm)

        # Create test documents
        doc = ParsedDocument(
            text="Machine learning is a subset of AI.",
            chunks=[
                "Machine learning is a subset of AI.",
                "Types include supervised and unsupervised learning.",
            ],
            metadata={"file_name": "test.txt", "title": "ML Basics"},
            file_path=tmp_path / "test.txt",
        )

        result = generator.generate(documents=[doc], num_examples=2)

        # Verify results
        assert len(result) == 2
        assert result[0]["question"] == "What is machine learning?"
        assert result[0]["answer"] == "A subset of AI."
        assert result[0]["difficulty"] == "easy"
        assert result[0]["type"] == "factual"
        assert result[0]["source_file"] == "test.txt"
        assert result[0]["source_metadata"]["file_name"] == "test.txt"
        assert "context" in result[0]

    def test_generate_with_markdown_code_blocks(self, tmp_path: Path) -> None:
        """Test handling of markdown code block in LLM response."""
        # Mock LLM that returns JSON in markdown code block
        mock_llm = MagicMock()
        qa_response = """```json
[
    {
        "question": "What is AI?",
        "answer": "Artificial Intelligence.",
        "difficulty": "easy",
        "type": "factual"
    }
]
```"""
        mock_llm.generate.return_value = qa_response

        generator = QAGenerator(llm=mock_llm)

        doc = ParsedDocument(
            text="AI is amazing.",
            chunks=["AI is amazing."],
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        result = generator.generate(documents=[doc], num_examples=1)

        assert len(result) == 1
        assert result[0]["question"] == "What is AI?"

    def test_generate_with_invalid_json(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of invalid JSON response."""
        # Mock LLM with invalid JSON
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "This is not valid JSON"

        generator = QAGenerator(llm=mock_llm)

        doc = ParsedDocument(
            text="Some text.",
            chunks=["Some text."],
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        with caplog.at_level("WARNING"):
            result = generator.generate(documents=[doc], num_examples=1)

        assert result == []
        assert "Failed to parse JSON response" in caplog.text

    def test_generate_with_llm_error(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of LLM generation errors."""
        # Mock LLM that raises an error
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        generator = QAGenerator(llm=mock_llm)

        doc = ParsedDocument(
            text="Some text.",
            chunks=["Some text."],
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        with caplog.at_level("ERROR"):
            result = generator.generate(documents=[doc], num_examples=1)

        # Should return empty or continue with other chunks
        assert "Error generating QA from chunk" in caplog.text

    def test_generate_multiple_documents(self, tmp_path: Path) -> None:
        """Test generation from multiple documents."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "question": "Test question?",
                "answer": "Test answer.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        generator = QAGenerator(llm=mock_llm)

        docs = [
            ParsedDocument(
                text="Doc 1 text",
                chunks=["Doc 1 chunk 1", "Doc 1 chunk 2"],
                metadata={"file_name": "doc1.txt"},
                file_path=tmp_path / "doc1.txt",
            ),
            ParsedDocument(
                text="Doc 2 text",
                chunks=["Doc 2 chunk 1", "Doc 2 chunk 2"],
                metadata={"file_name": "doc2.txt"},
                file_path=tmp_path / "doc2.txt",
            ),
        ]

        result = generator.generate(documents=docs, num_examples=4)

        # Should generate QA pairs from multiple documents
        assert len(result) <= 4
        assert all("question" in qa for qa in result)
        assert all("answer" in qa for qa in result)

    def test_generate_respects_num_examples(self, tmp_path: Path) -> None:
        """Test that generation respects num_examples limit."""
        mock_llm = MagicMock()
        # Return multiple QA pairs
        mock_llm.generate.return_value = json.dumps([
            {
                "question": f"Question {i}?",
                "answer": f"Answer {i}.",
                "difficulty": "easy",
                "type": "factual",
            }
            for i in range(5)  # Return 5 QA pairs per call
        ])

        generator = QAGenerator(llm=mock_llm)

        doc = ParsedDocument(
            text="Long document",
            chunks=[f"Chunk {i}" for i in range(10)],  # Many chunks
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        # Request only 3 examples
        result = generator.generate(documents=[doc], num_examples=3)

        # Should not exceed requested number
        assert len(result) <= 3

    def test_prompt_formatting(self, tmp_path: Path) -> None:
        """Test that prompts are formatted correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "question": "Test?",
                "answer": "Answer.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        generator = QAGenerator(
            llm=mock_llm,
            question_types=["factual"],
            difficulty_levels=["hard"],
            min_answer_length=10,
            max_answer_length=50,
        )

        doc = ParsedDocument(
            text="Test text",
            chunks=["Test chunk"],
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        generator.generate(documents=[doc], num_examples=1)

        # Check that generate was called with a properly formatted prompt
        call_args = mock_llm.generate.call_args[0][0]
        assert "Test chunk" in call_args
        assert "factual" in call_args
        assert "hard" in call_args or "easy" in call_args or "medium" in call_args
        assert "10" in call_args
        assert "50" in call_args

    def test_repr(self) -> None:
        """Test string representation."""
        mock_llm = MagicMock()
        mock_llm.__repr__ = MagicMock(return_value="MockLLM()")

        generator = QAGenerator(llm=mock_llm)
        repr_str = repr(generator)

        assert "QAGenerator" in repr_str
        assert "MockLLM" in repr_str

    def test_different_question_types_and_difficulties(self, tmp_path: Path) -> None:
        """Test that different question types and difficulties are used."""
        mock_llm = MagicMock()

        # Track the prompts that were generated
        prompts_generated = []

        def capture_prompt(prompt: str) -> str:
            prompts_generated.append(prompt)
            return json.dumps([
                {
                    "question": "Test question?",
                    "answer": "Test answer.",
                    "difficulty": "easy",
                    "type": "factual",
                }
            ])

        mock_llm.generate.side_effect = capture_prompt

        generator = QAGenerator(
            llm=mock_llm,
            question_types=["factual", "reasoning", "comparison"],
            difficulty_levels=["easy", "medium", "hard"],
        )

        doc = ParsedDocument(
            text="Test",
            chunks=[f"Chunk {i}" for i in range(10)],
            metadata={"file_name": "test.txt"},
            file_path=tmp_path / "test.txt",
        )

        generator.generate(documents=[doc], num_examples=10)

        # Should have called generate multiple times
        assert len(prompts_generated) > 0

        # Check that different types and difficulties appear in prompts
        all_prompts = " ".join(prompts_generated)
        assert any(qtype in all_prompts for qtype in ["factual", "reasoning", "comparison"])
        assert any(diff in all_prompts for diff in ["easy", "medium", "hard"])
