"""Integration tests for CorpusCraft pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpuscraft.config import ProcessingConfig
from corpuscraft.generators.qa_generator import QAGenerator
from corpuscraft.output.jsonl_writer import JSONLWriter
from corpuscraft.parsers.docling_parser import DoclingParser, ParsedDocument


class TestEndToEndPipeline:
    """Integration tests for the complete pipeline."""

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_full_qa_generation_pipeline(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test complete pipeline from parsing to output."""
        # Setup: Create test documents
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "doc1.txt").write_text(
            "Machine learning is a subset of AI. It enables computers to learn from data."
        )
        (doc_dir / "doc2.txt").write_text(
            "Deep learning uses neural networks. It is very powerful for image recognition."
        )

        # Mock document converter
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            content = Path(file_path).read_text()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = content
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        # Step 1: Parse documents
        processing_config = ProcessingConfig(chunk_size=50, chunk_overlap=10)
        parser = DoclingParser(processing_config)
        parser.converter = mock_converter

        parsed_docs = parser.parse_directory(doc_dir, file_types=["txt"])
        assert len(parsed_docs) == 2

        # Step 2: Generate QA pairs
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "question": "What is machine learning?",
                "answer": "A subset of AI that enables computers to learn from data.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        generator = QAGenerator(llm=mock_llm)
        qa_pairs = generator.generate(parsed_docs, num_examples=2)

        assert len(qa_pairs) > 0
        assert all("question" in qa for qa in qa_pairs)
        assert all("answer" in qa for qa in qa_pairs)

        # Step 3: Write to JSONL
        output_dir = tmp_path / "output"
        writer = JSONLWriter(output_dir=output_dir)
        output_files = writer.write(qa_pairs, dataset_name="qa_dataset")

        # Verify outputs
        assert "train" in output_files
        assert output_files["train"].exists()

        # Read and verify content
        train_data = JSONLWriter.read_jsonl(output_files["train"])
        assert len(train_data) > 0
        assert "question" in train_data[0]
        assert "answer" in train_data[0]

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_pipeline_with_multiple_generators(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test pipeline with multiple QA generators."""
        # Create test document
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "test.txt").write_text("Test content for multiple generators.")

        # Mock converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content for multiple generators."
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Parse
        parser = DoclingParser(ProcessingConfig())
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        # Create two different generators
        mock_llm1 = MagicMock()
        mock_llm1.generate.return_value = json.dumps([
            {
                "question": "Easy question?",
                "answer": "Easy answer.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        mock_llm2 = MagicMock()
        mock_llm2.generate.return_value = json.dumps([
            {
                "question": "Hard question?",
                "answer": "Complex answer.",
                "difficulty": "hard",
                "type": "reasoning",
            }
        ])

        generator1 = QAGenerator(llm=mock_llm1, difficulty_levels=["easy"])
        generator2 = QAGenerator(llm=mock_llm2, difficulty_levels=["hard"])

        # Generate with both
        qa_pairs1 = generator1.generate(docs, num_examples=1)
        qa_pairs2 = generator2.generate(docs, num_examples=1)

        # Combine results
        all_qa_pairs = qa_pairs1 + qa_pairs2

        # Write combined output
        writer = JSONLWriter(output_dir=tmp_path / "output")
        output_files = writer.write(all_qa_pairs, dataset_name="combined")

        assert len(output_files) > 0

    def test_pipeline_with_empty_documents(self, tmp_path: Path) -> None:
        """Test pipeline behavior with no documents."""
        mock_llm = MagicMock()
        generator = QAGenerator(llm=mock_llm)

        # Generate with empty document list
        qa_pairs = generator.generate([], num_examples=10)

        assert qa_pairs == []

        # Try to write empty results
        writer = JSONLWriter(output_dir=tmp_path / "output")
        output_files = writer.write(qa_pairs, dataset_name="empty")

        assert output_files == {}

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_pipeline_statistics_tracking(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that statistics are tracked throughout pipeline."""
        # Create documents
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        for i in range(5):
            (doc_dir / f"doc{i}.txt").write_text(f"Content {i} " * 100)

        # Mock converter
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            content = Path(file_path).read_text()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = content
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        # Parse and get statistics
        parser = DoclingParser(ProcessingConfig(chunk_size=100, chunk_overlap=10))
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        parse_stats = parser.get_statistics(docs)
        assert parse_stats["total_documents"] == 5
        assert parse_stats["total_chunks"] > 0

        # Generate QA pairs
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "question": "Q?",
                "answer": "A.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        generator = QAGenerator(llm=mock_llm)
        qa_pairs = generator.generate(docs, num_examples=10)

        # Write and get statistics
        writer = JSONLWriter(output_dir=tmp_path / "output")
        output_stats = writer.get_statistics(qa_pairs)

        assert output_stats["total_examples"] == len(qa_pairs)
        assert "fields" in output_stats

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_pipeline_reproducibility(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that pipeline produces reproducible results with same seed."""
        # Create test document
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "test.txt").write_text("Test content " * 50)

        # Mock converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content " * 50
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Parse
        parser = DoclingParser(ProcessingConfig())
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        # Generate with mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {"question": f"Q?", "answer": "A.", "difficulty": "easy", "type": "factual"}
        ])

        generator = QAGenerator(llm=mock_llm)
        qa_pairs = generator.generate(docs, num_examples=20)

        # Write with seed
        writer1 = JSONLWriter(
            output_dir=tmp_path / "run1",
            shuffle=True,
            seed=42,
        )
        files1 = writer1.write(qa_pairs, dataset_name="data")
        data1 = JSONLWriter.read_jsonl(files1["train"])

        # Write again with same seed
        writer2 = JSONLWriter(
            output_dir=tmp_path / "run2",
            shuffle=True,
            seed=42,
        )
        files2 = writer2.write(qa_pairs, dataset_name="data")
        data2 = JSONLWriter.read_jsonl(files2["train"])

        # Should be identical
        assert data1 == data2


class TestErrorHandling:
    """Integration tests for error handling."""

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_pipeline_continues_after_parsing_errors(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that pipeline continues when some files fail to parse."""
        # Create documents
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "good1.txt").write_text("Good content 1")
        (doc_dir / "bad.txt").write_text("Bad content")
        (doc_dir / "good2.txt").write_text("Good content 2")

        # Mock converter - fail on bad.txt
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            if "bad" in file_path:
                raise Exception("Parse error")
            content = Path(file_path).read_text()
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = content
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        # Parse - should skip bad file
        parser = DoclingParser(ProcessingConfig())
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        # Should have parsed 2 out of 3 files
        assert len(docs) == 2

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_pipeline_handles_generation_errors(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test pipeline when LLM generation fails."""
        # Create document
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "test.txt").write_text("Test content")

        # Mock converter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Parse
        parser = DoclingParser(ProcessingConfig())
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        # Mock LLM that fails
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        generator = QAGenerator(llm=mock_llm)

        # Should handle error gracefully
        qa_pairs = generator.generate(docs, num_examples=5)

        # May return empty or partial results
        assert isinstance(qa_pairs, list)


class TestDataFlow:
    """Test data flow through the pipeline."""

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_metadata_preservation(
        self, mock_converter_class: MagicMock, tmp_path: Path
    ) -> None:
        """Test that metadata is preserved through pipeline."""
        # Create document
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        test_file = doc_dir / "important.txt"
        test_file.write_text("Important content")

        # Mock converter with metadata
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Important content"
        mock_result.document.meta = MagicMock()
        mock_result.document.meta.title = "Important Document"
        mock_result.document.meta.authors = ["Author Name"]
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        # Parse
        parser = DoclingParser(ProcessingConfig())
        parser.converter = mock_converter
        docs = parser.parse_directory(doc_dir, file_types=["txt"])

        # Generate
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps([
            {
                "question": "What is this about?",
                "answer": "Important stuff.",
                "difficulty": "easy",
                "type": "factual",
            }
        ])

        generator = QAGenerator(llm=mock_llm)
        qa_pairs = generator.generate(docs, num_examples=1)

        # Verify metadata is in output
        assert len(qa_pairs) > 0
        assert qa_pairs[0]["source_file"] == "important.txt"
        assert "source_metadata" in qa_pairs[0]
        assert qa_pairs[0]["source_metadata"]["title"] == "Important Document"
        assert qa_pairs[0]["source_metadata"]["authors"] == ["Author Name"]
