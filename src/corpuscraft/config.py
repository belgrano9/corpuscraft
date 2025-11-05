"""Configuration models for CorpusCraft."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMBackend(str, Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class GeneratorType(str, Enum):
    """Supported generator types."""

    QA = "qa"
    EMBEDDING_PAIRS = "embedding_pairs"
    SUMMARY = "summary"
    CLASSIFICATION = "classification"


class OutputFormat(str, Enum):
    """Supported output formats."""

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"


class InputConfig(BaseModel):
    """Input configuration."""

    folder: Path = Field(description="Path to input documents")
    file_types: list[str] = Field(
        default=["pdf", "docx", "pptx", "html", "md", "txt"],
        description="File types to process",
    )
    recursive: bool = Field(default=True, description="Recursively search for files")


class ProcessingConfig(BaseModel):
    """Document processing configuration."""

    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    ocr_enabled: bool = Field(default=True, description="Enable OCR for scanned documents")
    extract_tables: bool = Field(default=True, description="Extract tables from documents")
    extract_images: bool = Field(default=False, description="Extract images from documents")


class LLMConfig(BaseModel):
    """LLM backend configuration."""

    backend: LLMBackend = Field(default=LLMBackend.OLLAMA, description="LLM backend to use")
    model: str = Field(default="llama3.1:8b", description="Model name/identifier")
    base_url: str | None = Field(
        default="http://localhost:11434", description="Base URL for API (Ollama, vLLM)"
    )
    api_key: str | None = Field(default=None, description="API key for cloud providers")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    device: str | None = Field(default=None, description="Device for local models (cuda, cpu)")
    batch_size: int = Field(default=1, description="Batch size for generation")


class QAGeneratorConfig(BaseModel):
    """QA generator specific configuration."""

    num_examples: int = Field(default=100, description="Number of QA pairs to generate")
    question_types: list[str] = Field(
        default=["factual", "reasoning", "comparison"],
        description="Types of questions to generate",
    )
    difficulty_levels: list[str] = Field(
        default=["easy", "medium", "hard"], description="Difficulty levels"
    )
    min_answer_length: int = Field(default=1, description="Minimum answer length in words")
    max_answer_length: int = Field(default=100, description="Maximum answer length in words")


class EmbeddingPairsConfig(BaseModel):
    """Embedding pairs generator specific configuration."""

    num_examples: int = Field(default=500, description="Number of query-passage pairs")
    include_hard_negatives: bool = Field(
        default=True, description="Include hard negative examples"
    )
    num_negatives: int = Field(default=3, description="Number of negative examples per query")


class GeneratorConfig(BaseModel):
    """Generator configuration."""

    type: GeneratorType = Field(description="Type of generator")
    qa: QAGeneratorConfig | None = Field(default=None, description="QA generator config")
    embedding_pairs: EmbeddingPairsConfig | None = Field(
        default=None, description="Embedding pairs config"
    )
    custom_params: dict[str, Any] = Field(
        default_factory=dict, description="Custom generator parameters"
    )


class OutputConfig(BaseModel):
    """Output configuration."""

    format: OutputFormat = Field(default=OutputFormat.JSONL, description="Output format")
    output_dir: Path = Field(default=Path("./outputs"), description="Output directory")
    split_ratio: list[float] = Field(
        default=[0.8, 0.1, 0.1], description="Train/val/test split ratios"
    )
    shuffle: bool = Field(default=True, description="Shuffle data before splitting")
    seed: int | None = Field(default=42, description="Random seed for reproducibility")


class CorpusCraftConfig(BaseSettings):
    """Main CorpusCraft configuration."""

    input: InputConfig = Field(description="Input configuration")
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Processing configuration"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    generators: list[GeneratorConfig] = Field(
        description="List of generators to run", min_length=1
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )

    class Config:
        """Pydantic config."""

        env_prefix = "CORPUSCRAFT_"
        case_sensitive = False
