# CorpusCraft

> Transform your documents into training datasets with local-first synthetic data generation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CorpusCraft** is a modular pipeline that transforms document collections (PDF, DOCX, PPTX, and more) into high-quality training datasets for tasks like question-answering, retrieval, and embeddings — running entirely on your machine.

## Features

- **Local-first**: No API costs, no data leaving your machine (Ollama backend)
- **Multiple parsers**: Docling (standard/GPU), OCR, VLM, DocLayout-YOLO, MinerU, PyMuPDF4LLM, and pdfplumber for versatile extraction
- **Consensus mode**: Ensemble processing that runs multiple parsers in parallel and merges outputs by bbox agreement
- **Flexible LLM backends**: Ollama, OpenAI, Anthropic
- **JSONL output** with configurable train/val/test splits

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/belgrano9/corpuscraft.git
cd corpuscraft

# Core install
uv sync

# With optional extras
uv sync --extra yolo        # DocLayout-YOLO parser
uv sync --extra cloud       # OpenAI / Anthropic backends
uv sync --extra all         # Everything
```

## Quick start

```bash
# 1. Initialize a project config
uv run corpuscraft init --input ./data --output ./outputs

# 2. Place PDFs in ./data, then generate
uv run corpuscraft generate --config corpuscraft_config.yaml
```

Output:
```
outputs/
  dataset_train.jsonl   # 80%
  dataset_val.jsonl     # 10%
  dataset_test.jsonl    # 10%
  dataset_metadata.json
```

## Parsers

| Key | Backend | Best for |
|-----|---------|----------|
| `standard` | Docling (CPU) | Native PDFs, default |
| `gpu` | Docling (CUDA) | Same, faster |
| `ocr` | Docling + RapidOCR | Scanned / image PDFs |
| `vlm` | Ollama vision LLM | Complex or ambiguous layouts |
| `yolo` | DocLayout-YOLO + pdfplumber | Fast, layout-aware text extraction |
| `mineru` | MinerU Pipeline | Multi-column, math formulas, 109-language OCR |
| `consensus` | Ensemble (Docling+YOLO+MinerU) | High-accuracy validation via bounding box agreement |
| `pymupdf` | PyMuPDF4LLM | Extremely fast, high-quality Markdown conversion |
| `pdfplumber`| pdfplumber | Lightweight text and basic table extraction |

Set `pipeline` in your config or pass `--pipeline yolo` on the CLI.

The YOLO parser uses [`juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained`](https://huggingface.co/juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained) (DocLayNet labels: Table, Picture, Caption, etc.). Requires the `yolo` extra.

## Configuration

```yaml
# corpuscraft_config.yaml
input_dir: ./data

parser:
  pipeline: standard          # standard | gpu | ocr | vlm | yolo | mineru | consensus | pymupdf | pdfplumber
  yolo_confidence: 0.2

llm:
  backend: ollama
  model: qwen3.5:2b
  base_url: http://localhost:11434
  temperature: 0.7

generators:
  - type: qa
    num_examples: 100

exporter:
  format: jsonl
  output_dir: ./outputs
  split_ratio: [0.8, 0.1, 0.1]
```

## Project structure

```
src/corpuscraft/
  parsers/
    base.py           # BaseParser ABC
    consensus.py      # Multi-parser ensembling
    mineru.py         # MinerU pipeline
    pdfplumber_parser.py # pdfplumber basic extraction
    pymupdf_parser.py # pymupdf4llm fast markdown
    standard.py       # Docling CPU/GPU
    ocr.py            # Docling + RapidOCR
    vlm.py            # Ollama VLM
    yolo.py           # DocLayout-YOLO + pdfplumber
    factory.py        # PipelineType → parser instance
  generators/
    qa.py             # Q&A pair generation
  exporters/
    jsonl.py          # JSONL export with splits
  models.py           # ParsedDocument dataclass
  config.py           # Pydantic config models
  cli.py              # Typer CLI
```

## Examples

All examples run with `uv run`:

```bash
uv run examples/basic/simple_processor.py
uv run examples/advanced/parser_comparison.py
```

| Script | Description |
|--------|-------------|
| `basic/simple_processor.py` | Basic Docling conversion |
| `basic/simple_processor_gpu.py` | GPU-accelerated Docling |
| `basic/pymupdf_example.py` | Fast PyMuPDF4LLM markdown conversion |
| `basic/pdfplumber_example.py` | Native pdfplumber layout extraction |
| `ocr/force_ocr.py` | Force OCR on a PDF |
| `ocr/translation_ocr.py` | OCR with translation |
| `vlm/test_vlm.py` | VLM processing via Ollama |
| `vlm/annotate_picture.py` | Annotate figures with VLM |
| `vlm/ollama_configs.py` | VLM configuration options |
| `advanced/figure_export.py` | Export page/table/figure images with Docling |
| `advanced/parser_comparison.py` | Side-by-side Docling vs YOLO detection overlay |
| `advanced/simple_yolo.py` | Minimal YOLO detection test on a single image or PDF |

## Development

```bash
uv sync --extra dev
uv run pytest tests/
uv run ruff check src/
uv run black src/
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Docling](https://github.com/DS4SD/docling) for document parsing
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for layout detection
- [Ollama](https://ollama.com) for local LLM inference
