# CorpusCraft — Claude Code Guide

## Project overview

CorpusCraft converts PDF documents into training datasets using local-first synthetic data generation. It supports multiple PDF parsing backends and generates Q&A pairs (and other formats) via Ollama or cloud LLMs.

## Running commands

Always use `uv run` — never invoke `.venv/Scripts/python.exe` directly.

```bash
uv run examples/basic/preprocess_then_parse.py
uv run examples/advanced/parser_comparison.py
```

There is no CLI. All modules are used directly as Python APIs.

## Architecture

```
src/corpuscraft/
  preprocessing/
    base.py          # BasePreprocessor ABC — run / run_folder
    poppler.py       # PopplerPreprocessor — inspect, probe, clean, split, rasterize
    models.py        # PreprocessedPDF + PDFMetadata dataclasses
  parsers/
    base.py          # BaseParser ABC — parse_file / parse_folder
    consensus.py     # Ensemble parser (Docling+YOLO+MinerU)
    mineru.py        # MinerU pipeline (math, multi-column, OCR)
    pdfplumber_parser.py # Lightweight pdfplumber extraction
    pymupdf_parser.py    # PyMuPDF4LLM for fast markdown conversion
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
  config.py          # Pydantic config (PreprocessingConfig, ParserConfig, LLMConfig, …)
```

## Preprocessing (poppler-utils)

`PopplerPreprocessor` runs before any parser. Requires poppler binaries on PATH
(Windows builds: https://github.com/oschwartz10612/poppler-windows/releases).

| Operation | Tool | What it does |
|---|---|---|
| inspect | `pdfinfo` | Page count, encryption, dimensions — always runs |
| probe | `pdftotext` | Detects native text vs scanned — always runs |
| clean | `pdftocairo` | Strips invisible text, annotations, comments (`clean=True`) |
| split | `pdfseparate` | One PDF per page (`split=True`) |
| rasterize | `pdftoppm` | Pages → PNG/JPEG at given DPI (`rasterize=True`) |

```python
from corpuscraft.preprocessing.poppler import PopplerPreprocessor

pre = PopplerPreprocessor(clean=True, split=False, rasterize=False)
result = pre.run(pdf_path, output_dir)

result.is_scanned          # True → use ocr/vlm pipeline
result.metadata.page_count
result.parser_input()      # cleaned PDF path (or original if clean=False)
result.page_images         # list of PNG paths (if rasterize=True)
result.page_pdfs           # list of per-page PDF paths (if split=True)
```

## Parser selection

| Pipeline | When to use |
|---|---|
| `standard` | Native PDFs, best quality |
| `gpu` | Same as standard but CUDA-accelerated |
| `ocr` | Scanned / image-only PDFs |
| `vlm` | Complex layouts via vision LLM |
| `yolo` | Layout-aware extraction, fastest |
| `mineru` | Multi-column, math formulas, dense OCR |
| `consensus` | High accuracy merging via bbox intersection |
| `pymupdf` | Extremely fast high-quality Markdown conversion |
| `pdfplumber` | Lightweight raw text and basic tables |

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
    figure_export.py          # Docling: extract page/table/figure images
    parser_comparison.py      # Side-by-side Docling vs YOLO detection overlay
    simple_consensus.py       # Runs multiple parsers and merges outputs
    simple_yolo.py            # Minimal YOLO test on a single image or PDF page
  basic/
    poppler_preprocess.py     # PopplerPreprocessor: all 5 operations, full report
    preprocess_then_parse.py  # Preprocess → auto-route → parse (full chain)
    pdfplumber_example.py     # Native pdfplumber layout extraction
    pymupdf_example.py        # Fast PyMuPDF4LLM markdown conversion
```

`parser_comparison.py` outputs per-page PNGs to `scratch/comparison/`. The `scratch/` directory is gitignored.
