from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    standard = "standard"
    gpu = "gpu"
    vlm = "vlm"
    ocr = "ocr"


class ParserConfig(BaseModel):
    pipeline: PipelineType = PipelineType.standard
    ocr_engine: str = "rapidocr"
    ocr_languages: list[str] = Field(default_factory=lambda: ["eng"])
    vlm_model: str = "gabegoodhart/granite-docling:258M"
    vlm_host: str = "http://localhost:11434"
    image_scale: float = 2.0


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7


class GeneratorConfig(BaseModel):
    type: str
    num_examples: int = 100


class ExporterConfig(BaseModel):
    format: str = "jsonl"
    output_dir: Path = Path("./outputs")
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1)


class CorpusCraftConfig(BaseModel):
    input_dir: Path
    parser: ParserConfig = Field(default_factory=ParserConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generators: list[GeneratorConfig] = Field(default_factory=list)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)


def load_config(path: Path) -> CorpusCraftConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return CorpusCraftConfig.model_validate(data)


def save_default_config(path: Path, input_dir: Path, output_dir: Path) -> None:
    config = {
        "input_dir": str(input_dir),
        "parser": {
            "pipeline": "standard",
            "ocr_engine": "rapidocr",
            "ocr_languages": ["eng"],
            "vlm_model": "gabegoodhart/granite-docling:258M",
            "vlm_host": "http://localhost:11434",
            "image_scale": 2.0,
        },
        "llm": {
            "backend": "ollama",
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
        },
        "generators": [
            {"type": "qa", "num_examples": 100},
        ],
        "exporter": {
            "format": "jsonl",
            "output_dir": str(output_dir),
            "split_ratio": [0.8, 0.1, 0.1],
        },
    }
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
