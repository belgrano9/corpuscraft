"""Tests for configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from corpuscraft.config import (
    CorpusCraftConfig,
    EmbeddingPairsConfig,
    GeneratorConfig,
    GeneratorType,
    InputConfig,
    LLMBackend,
    LLMConfig,
    OutputConfig,
    OutputFormat,
    ProcessingConfig,
    QAGeneratorConfig,
)


class TestInputConfig:
    """Tests for InputConfig."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test default values for InputConfig."""
        config = InputConfig(folder=tmp_path)

        assert config.folder == tmp_path
        assert config.file_types == ["pdf", "docx", "pptx", "html", "md", "txt"]
        assert config.recursive is True

    def test_custom_file_types(self, tmp_path: Path) -> None:
        """Test custom file types."""
        config = InputConfig(
            folder=tmp_path,
            file_types=["pdf", "txt"],
            recursive=False,
        )

        assert config.file_types == ["pdf", "txt"]
        assert config.recursive is False

    def test_folder_as_string(self) -> None:
        """Test that folder path can be provided as string."""
        config = InputConfig(folder="/tmp/test")
        assert isinstance(config.folder, Path)
        assert config.folder == Path("/tmp/test")


class TestProcessingConfig:
    """Tests for ProcessingConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ProcessingConfig()

        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.ocr_enabled is True
        assert config.extract_tables is True
        assert config.extract_images is False

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ProcessingConfig(
            chunk_size=1024,
            chunk_overlap=100,
            ocr_enabled=False,
        )

        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.ocr_enabled is False


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = LLMConfig()

        assert config.backend == LLMBackend.OLLAMA
        assert config.model == "llama3.1:8b"
        assert config.base_url == "http://localhost:11434"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        # Valid temperature
        config = LLMConfig(temperature=0.5)
        assert config.temperature == 0.5

        # Invalid temperature - too low
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-0.1)

        # Invalid temperature - too high
        with pytest.raises(ValidationError):
            LLMConfig(temperature=2.1)

    def test_different_backends(self) -> None:
        """Test different backend configurations."""
        # Ollama
        config_ollama = LLMConfig(backend=LLMBackend.OLLAMA)
        assert config_ollama.backend == LLMBackend.OLLAMA

        # OpenAI
        config_openai = LLMConfig(
            backend=LLMBackend.OPENAI,
            model="gpt-4",
            api_key="test-key",
        )
        assert config_openai.backend == LLMBackend.OPENAI
        assert config_openai.model == "gpt-4"
        assert config_openai.api_key == "test-key"


class TestQAGeneratorConfig:
    """Tests for QAGeneratorConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = QAGeneratorConfig()

        assert config.num_examples == 100
        assert config.question_types == ["factual", "reasoning", "comparison"]
        assert config.difficulty_levels == ["easy", "medium", "hard"]
        assert config.min_answer_length == 1
        assert config.max_answer_length == 100

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = QAGeneratorConfig(
            num_examples=50,
            question_types=["factual"],
            difficulty_levels=["easy"],
            min_answer_length=5,
            max_answer_length=50,
        )

        assert config.num_examples == 50
        assert config.question_types == ["factual"]
        assert config.difficulty_levels == ["easy"]
        assert config.min_answer_length == 5
        assert config.max_answer_length == 50


class TestEmbeddingPairsConfig:
    """Tests for EmbeddingPairsConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = EmbeddingPairsConfig()

        assert config.num_examples == 500
        assert config.include_hard_negatives is True
        assert config.num_negatives == 3

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = EmbeddingPairsConfig(
            num_examples=1000,
            include_hard_negatives=False,
            num_negatives=5,
        )

        assert config.num_examples == 1000
        assert config.include_hard_negatives is False
        assert config.num_negatives == 5


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_qa_generator_config(self) -> None:
        """Test QA generator configuration."""
        qa_config = QAGeneratorConfig(num_examples=50)
        gen_config = GeneratorConfig(
            type=GeneratorType.QA,
            qa=qa_config,
        )

        assert gen_config.type == GeneratorType.QA
        assert gen_config.qa == qa_config
        assert gen_config.qa.num_examples == 50

    def test_embedding_generator_config(self) -> None:
        """Test embedding pairs generator configuration."""
        emb_config = EmbeddingPairsConfig(num_examples=200)
        gen_config = GeneratorConfig(
            type=GeneratorType.EMBEDDING_PAIRS,
            embedding_pairs=emb_config,
        )

        assert gen_config.type == GeneratorType.EMBEDDING_PAIRS
        assert gen_config.embedding_pairs == emb_config

    def test_custom_params(self) -> None:
        """Test custom parameters."""
        gen_config = GeneratorConfig(
            type=GeneratorType.QA,
            custom_params={"custom_field": "value"},
        )

        assert gen_config.custom_params == {"custom_field": "value"}


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = OutputConfig()

        assert config.format == OutputFormat.JSONL
        assert config.output_dir == Path("./outputs")
        assert config.split_ratio == [0.8, 0.1, 0.1]
        assert config.shuffle is True
        assert config.seed == 42

    def test_custom_values(self, tmp_path: Path) -> None:
        """Test custom values."""
        config = OutputConfig(
            format=OutputFormat.CSV,
            output_dir=tmp_path,
            split_ratio=[0.7, 0.2, 0.1],
            shuffle=False,
            seed=123,
        )

        assert config.format == OutputFormat.CSV
        assert config.output_dir == tmp_path
        assert config.split_ratio == [0.7, 0.2, 0.1]
        assert config.shuffle is False
        assert config.seed == 123


class TestCorpusCraftConfig:
    """Tests for main CorpusCraftConfig."""

    def test_valid_config(self, tmp_path: Path) -> None:
        """Test valid complete configuration."""
        config = CorpusCraftConfig(
            input=InputConfig(folder=tmp_path),
            processing=ProcessingConfig(),
            llm=LLMConfig(),
            generators=[
                GeneratorConfig(
                    type=GeneratorType.QA,
                    qa=QAGeneratorConfig(num_examples=10),
                )
            ],
            output=OutputConfig(),
        )

        assert config.input.folder == tmp_path
        assert config.processing.chunk_size == 512
        assert config.llm.backend == LLMBackend.OLLAMA
        assert len(config.generators) == 1
        assert config.generators[0].type == GeneratorType.QA

    def test_minimal_config(self, tmp_path: Path) -> None:
        """Test minimal required configuration."""
        config = CorpusCraftConfig(
            input=InputConfig(folder=tmp_path),
            generators=[GeneratorConfig(type=GeneratorType.QA)],
        )

        # Check defaults are applied
        assert config.processing.chunk_size == 512
        assert config.llm.backend == LLMBackend.OLLAMA
        assert config.output.format == OutputFormat.JSONL

    def test_missing_required_fields(self) -> None:
        """Test that required fields are enforced."""
        # Missing input
        with pytest.raises(ValidationError):
            CorpusCraftConfig(
                generators=[GeneratorConfig(type=GeneratorType.QA)],
            )

        # Missing generators
        with pytest.raises(ValidationError):
            CorpusCraftConfig(
                input=InputConfig(folder="/tmp"),
                generators=[],
            )

    def test_multiple_generators(self, tmp_path: Path) -> None:
        """Test configuration with multiple generators."""
        config = CorpusCraftConfig(
            input=InputConfig(folder=tmp_path),
            generators=[
                GeneratorConfig(
                    type=GeneratorType.QA,
                    qa=QAGeneratorConfig(num_examples=50),
                ),
                GeneratorConfig(
                    type=GeneratorType.EMBEDDING_PAIRS,
                    embedding_pairs=EmbeddingPairsConfig(num_examples=100),
                ),
            ],
        )

        assert len(config.generators) == 2
        assert config.generators[0].type == GeneratorType.QA
        assert config.generators[1].type == GeneratorType.EMBEDDING_PAIRS


class TestEnums:
    """Tests for enum types."""

    def test_llm_backend_enum(self) -> None:
        """Test LLMBackend enum values."""
        assert LLMBackend.OLLAMA == "ollama"
        assert LLMBackend.OPENAI == "openai"
        assert LLMBackend.ANTHROPIC == "anthropic"
        assert LLMBackend.TRANSFORMERS == "transformers"
        assert LLMBackend.VLLM == "vllm"

    def test_generator_type_enum(self) -> None:
        """Test GeneratorType enum values."""
        assert GeneratorType.QA == "qa"
        assert GeneratorType.EMBEDDING_PAIRS == "embedding_pairs"
        assert GeneratorType.SUMMARY == "summary"
        assert GeneratorType.CLASSIFICATION == "classification"

    def test_output_format_enum(self) -> None:
        """Test OutputFormat enum values."""
        assert OutputFormat.JSONL == "jsonl"
        assert OutputFormat.CSV == "csv"
        assert OutputFormat.PARQUET == "parquet"
        assert OutputFormat.HUGGINGFACE == "huggingface"
