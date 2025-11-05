"""Tests for Docling document parser."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corpuscraft.config import ProcessingConfig
from corpuscraft.parsers.docling_parser import DoclingParser, ParsedDocument


class TestParsedDocument:
    """Tests for ParsedDocument class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test ParsedDocument initialization."""
        file_path = tmp_path / "test.txt"
        doc = ParsedDocument(
            text="Sample text",
            chunks=["Chunk 1", "Chunk 2"],
            metadata={"file_name": "test.txt"},
            file_path=file_path,
        )

        assert doc.text == "Sample text"
        assert doc.chunks == ["Chunk 1", "Chunk 2"]
        assert doc.metadata == {"file_name": "test.txt"}
        assert doc.file_path == file_path

    def test_repr(self, tmp_path: Path) -> None:
        """Test string representation."""
        file_path = tmp_path / "document.pdf"
        doc = ParsedDocument(
            text="A" * 1000,
            chunks=["chunk1", "chunk2", "chunk3"],
            metadata={},
            file_path=file_path,
        )

        repr_str = repr(doc)
        assert "document.pdf" in repr_str
        assert "chunks=3" in repr_str
        assert "chars=1000" in repr_str


class TestDoclingParser:
    """Tests for DoclingParser class."""

    def test_initialization(self) -> None:
        """Test DoclingParser initialization."""
        config = ProcessingConfig(
            chunk_size=512,
            chunk_overlap=50,
        )

        parser = DoclingParser(config)

        assert parser.config == config
        assert parser.text_splitter is not None
        assert parser.converter is not None

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_file_success(self, mock_converter_class: MagicMock, tmp_path: Path) -> None:
        """Test successful file parsing."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document for parsing.")

        # Mock DocumentConverter
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "This is a test document for parsing."
        mock_result.document.meta = MagicMock()
        mock_result.document.meta.title = "Test Document"
        mock_result.document.meta.authors = ["Test Author"]
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig(chunk_size=20, chunk_overlap=5)
        parser = DoclingParser(config)
        parser.converter = mock_converter

        # Parse the file
        parsed_doc = parser.parse_file(test_file)

        assert isinstance(parsed_doc, ParsedDocument)
        assert parsed_doc.text == "This is a test document for parsing."
        assert len(parsed_doc.chunks) > 0
        assert parsed_doc.metadata["file_name"] == "test.txt"
        assert parsed_doc.metadata["file_type"] == "txt"
        assert parsed_doc.metadata["title"] == "Test Document"
        assert parsed_doc.metadata["authors"] == ["Test Author"]
        assert parsed_doc.file_path == test_file

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_file_without_metadata(self, mock_converter_class: MagicMock, tmp_path: Path) -> None:
        """Test file parsing without metadata."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Simple test.")

        # Mock DocumentConverter without metadata
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Simple test."
        mock_result.document.meta = None  # No metadata
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig(chunk_size=100, chunk_overlap=10)
        parser = DoclingParser(config)
        parser.converter = mock_converter

        parsed_doc = parser.parse_file(test_file)

        assert parsed_doc.text == "Simple test."
        assert "file_name" in parsed_doc.metadata
        assert "title" not in parsed_doc.metadata
        assert "authors" not in parsed_doc.metadata

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_file_error(self, mock_converter_class: MagicMock, tmp_path: Path) -> None:
        """Test error handling during parsing."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock converter to raise an error
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("Parsing failed")
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig()
        parser = DoclingParser(config)
        parser.converter = mock_converter

        with pytest.raises(Exception, match="Parsing failed"):
            parser.parse_file(test_file)

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_directory(self, mock_converter_class: MagicMock, tmp_path: Path) -> None:
        """Test parsing directory."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("Document 1")
        (tmp_path / "doc2.txt").write_text("Document 2")
        (tmp_path / "doc3.pdf").write_text("Document 3")
        (tmp_path / "ignore.log").write_text("Should be ignored")

        # Mock converter
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = f"Content of {Path(file_path).name}"
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig()
        parser = DoclingParser(config)
        parser.converter = mock_converter

        # Parse directory
        parsed_docs = parser.parse_directory(
            tmp_path,
            file_types=["txt", "pdf"],
            recursive=False,
        )

        assert len(parsed_docs) == 3
        file_names = [doc.metadata["file_name"] for doc in parsed_docs]
        assert "doc1.txt" in file_names
        assert "doc2.txt" in file_names
        assert "doc3.pdf" in file_names
        assert "ignore.log" not in file_names

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_directory_recursive(self, mock_converter_class: MagicMock, tmp_path: Path) -> None:
        """Test recursive directory parsing."""
        # Create nested structure
        (tmp_path / "doc1.txt").write_text("Root doc")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc2.txt").write_text("Subdir doc")

        # Mock converter
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = f"Content of {Path(file_path).name}"
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig()
        parser = DoclingParser(config)
        parser.converter = mock_converter

        # Parse recursively
        parsed_docs = parser.parse_directory(
            tmp_path,
            file_types=["txt"],
            recursive=True,
        )

        assert len(parsed_docs) == 2

        # Parse non-recursively
        parsed_docs_non_recursive = parser.parse_directory(
            tmp_path,
            file_types=["txt"],
            recursive=False,
        )

        assert len(parsed_docs_non_recursive) == 1

    @patch("corpuscraft.parsers.docling_parser.DocumentConverter")
    def test_parse_directory_with_errors(
        self, mock_converter_class: MagicMock, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test directory parsing with some failed files."""
        # Create test files
        (tmp_path / "good.txt").write_text("Good file")
        (tmp_path / "bad.txt").write_text("Bad file")

        # Mock converter - one success, one failure
        mock_converter = MagicMock()

        def mock_convert(file_path: str) -> MagicMock:
            if "bad" in file_path:
                raise Exception("Failed to parse bad.txt")
            mock_result = MagicMock()
            mock_result.document.export_to_markdown.return_value = "Good content"
            return mock_result

        mock_converter.convert.side_effect = mock_convert
        mock_converter_class.return_value = mock_converter

        config = ProcessingConfig()
        parser = DoclingParser(config)
        parser.converter = mock_converter

        with caplog.at_level("WARNING"):
            parsed_docs = parser.parse_directory(tmp_path, file_types=["txt"])

        # Should only parse the good file
        assert len(parsed_docs) == 1
        assert parsed_docs[0].metadata["file_name"] == "good.txt"
        assert "Skipping" in caplog.text

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Test statistics generation."""
        docs = [
            ParsedDocument(
                text="A" * 100,
                chunks=["chunk1", "chunk2"],
                metadata={"file_type": "pdf"},
                file_path=tmp_path / "doc1.pdf",
            ),
            ParsedDocument(
                text="B" * 200,
                chunks=["chunk1", "chunk2", "chunk3"],
                metadata={"file_type": "txt"},
                file_path=tmp_path / "doc2.txt",
            ),
            ParsedDocument(
                text="C" * 150,
                chunks=["chunk1"],
                metadata={"file_type": "pdf"},
                file_path=tmp_path / "doc3.pdf",
            ),
        ]

        config = ProcessingConfig()
        parser = DoclingParser(config)
        stats = parser.get_statistics(docs)

        assert stats["total_documents"] == 3
        assert stats["total_characters"] == 450  # 100 + 200 + 150
        assert stats["total_chunks"] == 6  # 2 + 3 + 1
        assert stats["avg_chunks_per_doc"] == 2.0
        assert stats["file_types"] == {"pdf": 2, "txt": 1}

    def test_get_statistics_empty(self) -> None:
        """Test statistics with empty document list."""
        config = ProcessingConfig()
        parser = DoclingParser(config)
        stats = parser.get_statistics([])

        assert stats["total_documents"] == 0
        assert stats["avg_chunks_per_doc"] == 0

    def test_chunk_size_configuration(self, tmp_path: Path) -> None:
        """Test that chunk size is properly configured."""
        config1 = ProcessingConfig(chunk_size=100, chunk_overlap=10)
        parser1 = DoclingParser(config1)
        assert parser1.text_splitter._chunk_size == 100
        assert parser1.text_splitter._chunk_overlap == 10

        config2 = ProcessingConfig(chunk_size=500, chunk_overlap=50)
        parser2 = DoclingParser(config2)
        assert parser2.text_splitter._chunk_size == 500
        assert parser2.text_splitter._chunk_overlap == 50
