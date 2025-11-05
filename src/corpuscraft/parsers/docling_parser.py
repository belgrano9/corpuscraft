"""Docling-based document parser."""

import logging
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpuscraft.config import ProcessingConfig

logger = logging.getLogger(__name__)


class ParsedDocument:
    """Represents a parsed document with metadata."""

    def __init__(
        self,
        text: str,
        chunks: list[str],
        metadata: dict[str, Any],
        file_path: Path,
    ) -> None:
        """Initialize parsed document.

        Args:
            text: Full document text
            chunks: Document split into chunks
            metadata: Document metadata
            file_path: Original file path
        """
        self.text = text
        self.chunks = chunks
        self.metadata = metadata
        self.file_path = file_path

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParsedDocument(file={self.file_path.name}, "
            f"chunks={len(self.chunks)}, "
            f"chars={len(self.text)})"
        )


class DoclingParser:
    """Parser using Docling for document processing."""

    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize Docling parser.

        Args:
            config: Processing configuration
        """
        self.config = config
        self.converter = DocumentConverter()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized DoclingParser with chunk_size={config.chunk_size}, "
            f"overlap={config.chunk_overlap}"
        )

    def parse_file(self, file_path: Path) -> ParsedDocument:
        """Parse a single file.

        Args:
            file_path: Path to the file

        Returns:
            Parsed document

        Raises:
            Exception: If parsing fails
        """
        logger.info(f"Parsing file: {file_path}")

        try:
            # Convert document using Docling
            result = self.converter.convert(str(file_path))

            # Extract text content
            text = result.document.export_to_markdown()

            # Extract metadata
            metadata = {
                "file_name": file_path.name,
                "file_type": file_path.suffix.lstrip("."),
                "file_path": str(file_path),
            }

            # Add additional metadata if available
            if hasattr(result.document, "meta"):
                if hasattr(result.document.meta, "title"):
                    metadata["title"] = result.document.meta.title
                if hasattr(result.document.meta, "authors"):
                    metadata["authors"] = result.document.meta.authors

            # Split into chunks
            chunks = self.text_splitter.split_text(text)

            logger.info(
                f"Successfully parsed {file_path.name}: "
                f"{len(text)} chars, {len(chunks)} chunks"
            )

            return ParsedDocument(
                text=text,
                chunks=chunks,
                metadata=metadata,
                file_path=file_path,
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            raise

    def parse_directory(
        self,
        directory: Path,
        file_types: list[str] | None = None,
        recursive: bool = True,
    ) -> list[ParsedDocument]:
        """Parse all documents in a directory.

        Args:
            directory: Directory path
            file_types: List of file extensions to include (e.g., ['pdf', 'docx'])
            recursive: Whether to search recursively

        Returns:
            List of parsed documents
        """
        if file_types is None:
            file_types = ["pdf", "docx", "pptx", "html", "md", "txt"]

        logger.info(f"Scanning directory: {directory}")

        # Collect files
        files: list[Path] = []
        for ext in file_types:
            pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
            files.extend(directory.glob(pattern))

        logger.info(f"Found {len(files)} files to parse")

        # Parse each file
        parsed_docs: list[ParsedDocument] = []
        for file_path in files:
            try:
                doc = self.parse_file(file_path)
                parsed_docs.append(doc)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

        logger.info(f"Successfully parsed {len(parsed_docs)}/{len(files)} files")

        return parsed_docs

    def get_statistics(self, documents: list[ParsedDocument]) -> dict[str, Any]:
        """Get statistics about parsed documents.

        Args:
            documents: List of parsed documents

        Returns:
            Dictionary of statistics
        """
        total_chars = sum(len(doc.text) for doc in documents)
        total_chunks = sum(len(doc.chunks) for doc in documents)

        file_types: dict[str, int] = {}
        for doc in documents:
            ext = doc.metadata["file_type"]
            file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / len(documents) if documents else 0,
            "file_types": file_types,
        }
