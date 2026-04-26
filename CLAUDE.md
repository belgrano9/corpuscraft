# CorpusCraft — Claude Code Guide

## Project overview

CorpusCraft converts PDF documents into training datasets using local-first synthetic data generation. It supports multiple PDF parsing backends and generates Q&A pairs (and other formats) via Ollama or cloud LLMs.

## Running commands

Always use `uv run` — never invoke `.venv/Scripts/python.exe` directly.

```bash
uv run examples/advanced/parser_comparison.py
uv run -m corpuscraft --help
```

## Architecture

```
src/corpuscraft/
  parsers/
    base.py          # BaseParser ABC — parse_file / parse_folder
    standard.py      # Docling (CPU) — default pipeline
    ocr.py           # Docling + RapidOCR — for scanned PDFs
    vlm.py           # Ollama VLM backend (granite-docling)
    yolo.py          # DocLayout-YOLO + pdfplumber — layout-aware text extraction
    factory.py       # Parser factory: PipelineType → parser instance
  generators/
    base.py
    qa.py            # Q&A pair generation via LLM
  exporters/
    jsonl.py         # JSONL export with train/val/test split
  models.py          # ParsedDocument dataclass
  config.py          # Pydantic config (ParserConfig, LLMConfig, …)
  cli.py             # Typer CLI entrypoint
```

## Parser selection

| Pipeline | Config key | When to use |
|---|---|---|
| `standard` | `pipeline: standard` | Native PDFs, best quality |
| `gpu` | `pipeline: gpu` | Same as standard but CUDA-accelerated |
| `ocr` | `pipeline: ocr` | Scanned / image-only PDFs |
| `vlm` | `pipeline: vlm` | Complex layouts via vision LLM |
| `yolo` | `pipeline: yolo` | Layout-aware extraction, fastest |

## YOLO parser notes

- Default model: `juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained`
  — uses DocLayNet labels (Table, Picture, Caption, Section-header, etc.)
  — **not** `DocStructBench` which uses different labels (abandon, plain_text) and is unsuitable for table/figure detection
- Renders pages with pypdfium2 at 150 DPI (`scale = 150/72`)
- Native PDFs only — scanned pages yield no text; use `ocr` pipeline instead
- Reading order is naive (top→bottom, left→right); 2-column layouts may interleave

## PyPI index strategy

`pyproject.toml` sets `index-strategy = "unsafe-best-match"` to prevent the PyTorch CUDA index from shadowing newer versions of packages like `requests` on PyPI.

## Examples

```
examples/
  advanced/
    figure_export.py        # Docling: extract page/table/figure images
    parser_comparison.py    # Side-by-side Docling vs YOLO detection overlay
    simple_yolo.py          # Minimal YOLO test on a single image or PDF page
```

`parser_comparison.py` outputs per-page PNGs to `scratch/comparison/`. The `scratch/` directory is gitignored.
