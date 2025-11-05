# CorpusCraft: Synthetic Dataset Generation Pipeline

## Problem Statement

Machine learning practitioners often have domain-specific documents (PDFs, DOCX, PPTX, etc.) but lack labeled datasets for training models like:
- Question-Answering (QA) systems
- Embedding models
- Retrieval systems
- Summarization models
- Classification models

Creating high-quality training data manually is expensive and time-consuming. This project aims to automate the generation of diverse, synthetic datasets from existing document collections.

## Core Concept

**CorpusCraft** is a modular pipeline that transforms static document collections into diverse training datasets through:

1. **Document Ingestion**: Parse various document formats (PDF, DOCX, PPTX, TXT, MD, etc.)
2. **Content Processing**: Extract and structure text, tables, images, and metadata
3. **Synthetic Data Generation**: Use LLMs to generate task-specific training examples
4. **Quality Control**: Validate and filter generated examples
5. **Export**: Output in standard ML training formats

## Use Cases

### 1. Question-Answering Datasets
- **Input**: Technical documentation, manuals, research papers
- **Output**: Question-answer pairs with context passages
- **Formats**: SQuAD, Natural Questions, custom formats

### 2. Embedding Model Training
- **Input**: Domain-specific documents
- **Output**: Query-passage pairs, positive/negative examples
- **Formats**: Triplets (anchor, positive, negative), pair format

### 3. Retrieval/RAG Datasets
- **Input**: Knowledge base documents
- **Output**: Queries with relevant document IDs/passages
- **Formats**: BEIR, MS MARCO style

### 4. Summarization Datasets
- **Input**: Long-form content
- **Output**: Document-summary pairs at various granularities
- **Formats**: CNN/DailyMail, XSum style

### 5. Classification/Labeling
- **Input**: Mixed document types
- **Output**: Document-label pairs, hierarchical categories
- **Formats**: CSV, JSONL with labels

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                             │
│  [Folder with PDFs, DOCX, PPTX, TXT, MD, HTML, etc.]       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 DOCUMENT PARSERS                            │
│  Powered by Docling (IBM's enterprise doc parser)          │
│  • PDF (with OCR support)                                  │
│  • DOCX, PPTX (native parsing)                             │
│  • HTML, Markdown, Text                                     │
│  • Advanced: Tables, images, formulas, layout detection    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              CONTENT STRUCTURING                            │
│  • Text extraction & chunking                              │
│  • Metadata extraction (title, author, date)               │
│  • Table detection & extraction                            │
│  • Section/hierarchy detection                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           SYNTHETIC DATA GENERATORS                         │
│  ┌──────────────────────────────────────────┐              │
│  │  Task-Specific Generators:               │              │
│  │  • QA Generator (extractive/abstractive) │              │
│  │  • Query Generator (for embeddings)      │              │
│  │  • Summary Generator                     │              │
│  │  • Classification Label Generator        │              │
│  │  • Paraphrase Generator                  │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  LLM Backend (LOCAL-FIRST, configurable):                  │
│  PRIMARY: Local models                                     │
│  • Ollama (Llama 3, Mistral, Qwen, Gemma)                  │
│  • vLLM (optimized local inference)                        │
│  • HuggingFace Transformers (any OSS model)                │
│  OPTIONAL: Cloud APIs (OpenAI, Anthropic, Cohere)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              QUALITY CONTROL                                │
│  • Diversity scoring (avoid duplicates)                    │
│  • Relevance filtering                                     │
│  • Answer verification (for QA)                            │
│  • Length/format validation                                │
│  • Human-in-the-loop review interface (optional)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 OUTPUT LAYER                                │
│  • JSONL (standard ML format)                              │
│  • CSV (tabular data)                                      │
│  • HuggingFace Datasets format                             │
│  • Custom schemas                                          │
│  • Train/validation/test splits                            │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Local-First Design
- **Privacy**: All processing can happen offline - your documents never leave your machine
- **Cost**: Zero API costs when using local models
- **Control**: Full control over model selection and behavior
- **Speed**: No network latency, batch processing at your hardware limits

### 2. Modular Pipeline
- Each stage is independent and configurable
- Docling handles all document formats uniformly
- Pluggable LLM backends (local or cloud)
- Custom generator modules

### 3. Configuration-Driven
```yaml
input:
  folder: "./documents"
  file_types: ["pdf", "docx", "pptx", "html", "md"]

processing:
  chunk_size: 512
  chunk_overlap: 50
  docling:
    ocr_enabled: true
    extract_tables: true
    extract_images: false

llm:
  backend: "ollama"  # or "vllm", "transformers", "openai", "anthropic"
  model: "llama3.1:8b"
  base_url: "http://localhost:11434"  # for Ollama
  temperature: 0.7

  # Alternative: local transformers
  # backend: "transformers"
  # model: "meta-llama/Llama-3.1-8B-Instruct"
  # device: "cuda"

generators:
  - type: "qa"
    num_examples: 100
    question_types: ["factual", "reasoning", "comparison"]
    difficulty_levels: ["easy", "medium", "hard"]

  - type: "embedding_pairs"
    num_examples: 500
    include_hard_negatives: true

output:
  format: "jsonl"
  split_ratio: [0.8, 0.1, 0.1]  # train/val/test
```

### 3. Diverse Task Types

**QA Generation:**
- Extractive questions (answer is verbatim from text)
- Abstractive questions (requires synthesis)
- Multi-hop reasoning questions
- Clarification questions

**Embedding/Retrieval:**
- Query-passage pairs
- Hard negatives mining
- Semantic similarity pairs

**Data Augmentation:**
- Paraphrasing
- Back-translation
- Contextual word replacement

### 4. Quality Assurance
- Automatic quality scoring
- Deduplication
- Diversity metrics
- Optional human review interface

### 5. Scalability
- Batch processing
- Parallel generation
- Resume from checkpoint
- Progress tracking

## Technology Stack

### Core Libraries
- **Document Processing**: `docling` (unified parser for all formats)
  - Handles PDF, DOCX, PPTX, HTML, Markdown with layout analysis
  - Built-in OCR, table extraction, formula detection
  - Maintains document structure and hierarchy
- **Text Processing**: `langchain` for chunking strategies
- **LLM Integration (Local-First)**:
  - `ollama-python` (primary: Ollama integration)
  - `vllm` (optimized local inference)
  - `transformers` + `accelerate` (HuggingFace models)
  - Optional: `openai`, `anthropic`, `cohere` (cloud APIs)
- **Data Handling**: `pandas`, `datasets` (HuggingFace)

### Configuration
- **Config Management**: `pydantic` (validation), `pyyaml` (config files)
- **CLI**: `typer` (modern CLI with type hints)

### Optional
- **Vector DB**: `chromadb` (lightweight, local-first) or `faiss` for similarity search
- **Web UI**: `gradio` for human review and monitoring
- **Caching**: `diskcache` for parsed documents and generated examples

## Implementation Phases

### Phase 1: MVP (Core Pipeline)
- [ ] Docling integration for document parsing (all formats)
- [ ] Simple chunking strategy
- [ ] Ollama integration for local LLM inference
- [ ] One generator: QA pairs using local models
- [ ] JSONL output format
- [ ] CLI interface with typer

### Phase 2: Extensibility
- [ ] Advanced Docling features (tables, OCR, layout detection)
- [ ] Configurable chunking strategies
- [ ] Additional LLM backends:
  - [ ] vLLM for optimized inference
  - [ ] HuggingFace Transformers
  - [ ] Optional cloud APIs (OpenAI, Anthropic)
- [ ] Additional generators: embedding pairs, summaries, classification
- [ ] Multiple output formats (CSV, Parquet, HF Datasets)

### Phase 3: Quality & Scale
- [ ] Quality scoring and filtering
- [ ] Deduplication
- [ ] Batch processing and parallelization
- [ ] Progress tracking and checkpointing
- [ ] Train/val/test splitting

### Phase 4: Advanced Features
- [ ] Web UI for human review
- [ ] Hard negatives mining for embedding tasks
- [ ] Multi-modal support (images from PDFs)
- [ ] Active learning loop (iterative improvement)
- [ ] Pre-built templates for common domains (legal, medical, technical)

## Example Workflow

```bash
# 1. Start Ollama (if not running)
ollama pull llama3.1:8b

# 2. Initialize project with documents
corpuscraft init --input ./my_documents

# 3. Configure generation (creates config.yaml)
corpuscraft configure \
  --task qa \
  --backend ollama \
  --model llama3.1:8b \
  --num-examples 500

# 4. Run pipeline (100% local, no API calls)
corpuscraft generate --config config.yaml --output ./dataset

# 5. Validate results
corpuscraft validate --input ./dataset --metrics diversity,relevance

# 6. Export to HuggingFace
corpuscraft export --format huggingface --repo my-org/my-dataset

# Alternative: Use cloud API if needed
corpuscraft generate \
  --config config.yaml \
  --backend openai \
  --model gpt-4 \
  --output ./dataset
```

## Success Metrics

- **Coverage**: % of input documents represented in output dataset
- **Diversity**: Unique question types, topics, difficulty levels
- **Quality**: Human evaluation scores, automatic validation pass rate
- **Efficiency**: Time and cost to generate N examples
- **Usability**: Community adoption, GitHub stars, documentation quality

## Competitive Landscape

**Similar Projects:**
- `gpt-dataset-generator`: Focused on OpenAI, less modular
- `qa-gen`: QA-specific, not extensible to other tasks
- `synthetic-data-vault`: Tabular data focus

**Differentiation:**
- **Local-first architecture**: Privacy-focused, zero API costs
- **Docling-powered**: Best-in-class document parsing with layout understanding
- Multi-task support (QA, embeddings, retrieval, summarization)
- Document-format agnostic (PDF, DOCX, PPTX, HTML, MD)
- LLM-agnostic (works with any local or cloud model)
- Production-ready quality controls
- Open-source and community-driven

## Future Vision

- Pre-trained task templates for common domains
- Automated dataset versioning and lineage tracking
- Integration with training pipelines (auto-train on generated data)
- Marketplace for custom generators
- Support for multi-modal datasets (text + images)

---

**Project Name**: CorpusCraft
**Tagline**: "Transform your documents into training data"
**License**: MIT/Apache 2.0
**Language**: Python 3.10+
