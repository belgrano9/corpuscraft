# CorpusCraft Examples

This directory contains example scripts demonstrating various document processing capabilities using the Docling library. These examples serve as starting points for understanding how to process documents in different ways.

## Table of Contents

- [Quick Start](#quick-start)
- [Example Categories](#example-categories)
- [Examples Index](#examples-index)
- [Common Prerequisites](#common-prerequisites)
- [Tips & Best Practices](#tips--best-practices)

## Quick Start

1. **Basic Document Processing**: Start with `basic/simple_processor.py` for the simplest example
2. **VLM Processing**: Try `vlm/test_vlm.py` if you have Ollama installed
3. **OCR**: Use `ocr/force_ocr.py` to compare different OCR engines
4. **Advanced Export**: Run `advanced/figure_export.py` for comprehensive document export

## Example Categories

### 📄 Basic Examples (`basic/`)

Simple examples demonstrating core document processing functionality.

### 🤖 VLM Examples (`vlm/`)

Examples using Vision Language Models (VLM) with Ollama for advanced document understanding.

### 🔍 OCR Examples (`ocr/`)

Examples focused on Optical Character Recognition for scanned or image-based documents.

### 🚀 Advanced Examples (`advanced/`)

Complex examples demonstrating full document processing pipelines with comprehensive exports.

---

## Examples Index

| Example | Category | Complexity | Prerequisites | Description |
|---------|----------|------------|---------------|-------------|
| [simple_processor.py](#simple_processorpy) | Basic | ⭐ Beginner | Docling | Minimal PDF to Markdown conversion |
| [simple_processor_gpu.py](#simple_processor_gpupy) | Basic | ⭐⭐ Intermediate | Docling, CUDA | GPU-accelerated document processing |
| [ollama_configs.py](#ollama_configspy) | VLM | ⭐⭐ Intermediate | Docling, Ollama | Reusable VLM configuration module |
| [test_vlm.py](#test_vlmpy) | VLM | ⭐⭐ Intermediate | Docling, Ollama | Test VLM document processing |
| [annotate_picture.py](#annotate_picturepy) | VLM | ⭐⭐⭐ Advanced | Docling, Ollama | Extract picture annotations with VLM |
| [translation_ocr.py](#translation_ocrpy) | OCR | ⭐⭐ Intermediate | Docling, Tesseract | Multi-language OCR (French example) |
| [force_ocr.py](#force_ocrpy) | OCR | ⭐⭐ Intermediate | Docling, OCR engines | Compare different OCR engines |
| [figure_export.py](#figure_exportpy) | Advanced | ⭐⭐⭐ Advanced | Docling, PIL | Comprehensive figure/table export |

---

## Detailed Example Descriptions

### Basic Examples

#### `simple_processor.py`

**Purpose**: The simplest possible example of document processing with Docling.

**What it does**:
- Converts a PDF from a URL to Markdown
- Uses default DocumentConverter settings
- Prints output to console

**When to use**:
- Getting started with Docling
- Quick proof-of-concept
- Understanding basic conversion flow

**Usage**:
```bash
cd examples/basic
python simple_processor.py
```

**Key Features**:
- Minimal code (< 50 lines)
- No configuration required
- Good for learning the basics

---

#### `simple_processor_gpu.py`

**Purpose**: Demonstrate GPU acceleration for faster document processing.

**What it does**:
- Uses CUDA for accelerated processing
- Configures batch sizes for optimal GPU utilization
- Saves output to a Markdown file

**When to use**:
- Processing large documents
- Batch processing multiple documents
- You have an NVIDIA GPU available

**Usage**:
```bash
cd examples/basic
python simple_processor_gpu.py
```

**Key Features**:
- GPU acceleration with CUDA
- Configurable batch sizes
- Significantly faster for large documents

**Prerequisites**:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- Appropriate GPU drivers

---

### VLM Examples

#### `ollama_configs.py`

**Purpose**: Reusable configuration module for Ollama VLM processing.

**What it does**:
- Provides a pre-configured DocumentConverter for Ollama
- Supports both PDF and image inputs
- Includes connection checking utility

**When to use**:
- Building applications that need Ollama VLM
- Creating consistent VLM configurations
- Testing Ollama connectivity

**Usage**:
```python
from ollama_configs import converter, check_ollama_connection

if check_ollama_connection():
    result = converter.convert("document.pdf")
    print(result.document.export_to_markdown())
```

Or run directly to test configuration:
```bash
cd examples/vlm
python ollama_configs.py
```

**Key Features**:
- Environment variable support (`OLLAMA_HOST`, `OLLAMA_MODEL`)
- Connection checking
- Uses Granite Document OCR model
- DOCTAGS response format for better structure

**Environment Variables**:
- `OLLAMA_HOST`: Override Ollama endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Override model (default: gabegoodhart/granite-docling:258M)

---

#### `test_vlm.py`

**Purpose**: Simple test script for VLM document processing.

**What it does**:
- Imports configuration from `ollama_configs.py`
- Checks Ollama connectivity before processing
- Processes a test PDF and outputs Markdown

**When to use**:
- Testing your Ollama setup
- Verifying VLM models are working
- Quick VLM processing tests

**Usage**:
```bash
cd examples/vlm
python test_vlm.py
```

**Key Features**:
- Pre-flight connectivity check
- Helpful error messages if Ollama isn't available
- Formatted output with separators

**Required Setup**:
1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull gabegoodhart/granite-docling:258M`
3. Ensure Ollama is running

---

#### `annotate_picture.py`

**Purpose**: Extract and annotate picture elements from documents using VLM.

**What it does**:
- Processes documents with IBM Granite Vision model
- Extracts metadata about images, figures, and diagrams
- Prints captions and annotations for each picture

**When to use**:
- Extracting visual content metadata
- Understanding image/figure content
- Building image annotation datasets

**Usage**:
```bash
cd examples/vlm
python annotate_picture.py
```

**Key Features**:
- Uses specialized vision model (Granite Vision 3.3)
- Extracts picture captions and annotations
- Supports both PDF and image inputs
- Environment variable configuration

**Environment Variables**:
- `OLLAMA_HOST`: Override Ollama endpoint
- `VISION_MODEL`: Override vision model (default: ibm/granite3.3-vision:2b)

**Required Setup**:
1. Install Ollama
2. Pull the vision model: `ollama pull ibm/granite3.3-vision:2b`

---

### OCR Examples

#### `translation_ocr.py`

**Purpose**: Process scanned documents with multi-language OCR support.

**What it does**:
- Uses Tesseract OCR with French language pack
- Forces full-page OCR on all pages
- Exports to HTML with referenced images

**When to use**:
- Processing scanned documents
- Documents in languages other than English
- Image-based PDFs without embedded text

**Usage**:
```bash
cd examples/ocr
python translation_ocr.py
```

**Key Features**:
- Multi-language support (configured for French)
- Forced full-page OCR
- HTML export with images

**Language Configuration**:
```python
# French
ocr_options = TesseractCliOcrOptions(lang=["fra"])

# English
ocr_options = TesseractCliOcrOptions(lang=["eng"])

# Auto-detect
ocr_options = TesseractCliOcrOptions(lang=["auto"])

# Multiple languages
ocr_options = TesseractCliOcrOptions(lang=["fra", "eng"])
```

**Prerequisites**:
- Tesseract OCR installed
- Language packs: `apt install tesseract-ocr-fra` (Linux)

---

#### `force_ocr.py`

**Purpose**: Compare different OCR engines on the same document.

**What it does**:
- Demonstrates 5 different OCR engine options
- Forces full-page OCR with table structure detection
- Outputs Markdown to console

**When to use**:
- Testing which OCR engine works best for your documents
- Comparing OCR quality
- Evaluating OCR performance

**Usage**:
```bash
cd examples/ocr
python force_ocr.py
```

**Supported OCR Engines**:

1. **RapidOCR** (default) - Fast, lightweight, pure Python
   - Install: `pip install rapidocr-onnxruntime`
   - Best for: Quick processing, no dependencies

2. **TesseractCLI** - Industry standard
   - Requires Tesseract installation
   - Best for: High accuracy, language support

3. **Tesseract** - Python bindings
   - Install: `pip install pytesseract`
   - Best for: Same as CLI, Python integration

4. **EasyOCR** - Deep learning-based
   - Install: `pip install easyocr`
   - Best for: Complex layouts, handwritten text

5. **OcrMac** - macOS native
   - Built into macOS 10.15+
   - Best for: macOS users, no installation needed

**To switch engines**, uncomment the desired option in the script.

---

### Advanced Examples

#### `figure_export.py`

**Purpose**: Complete document processing pipeline with comprehensive export.

**What it does**:
- Extracts individual page images
- Exports table and figure images separately
- Generates multiple output formats (MD, HTML)
- Includes performance metrics

**When to use**:
- Creating datasets from documents
- Extracting all visual content
- Generating multiple output formats
- Building document processing pipelines

**Usage**:
```bash
cd examples/advanced
python figure_export.py
```

**Output Files** (saved to `scratch/` directory):
- `{filename}-page-{n}.png`: Individual page images
- `{filename}-table-{n}.png`: Extracted table images
- `{filename}-picture-{n}.png`: Extracted figure images
- `{filename}-with-images.md`: Markdown with embedded (base64) images
- `{filename}-with-image-refs.md`: Markdown with image file references
- `{filename}-with-image-refs.html`: HTML with image file references

**Key Features**:
- Configurable image resolution (default: 2x scale)
- Separate counters for tables vs pictures
- Multiple export formats
- Performance timing
- Comprehensive logging

**Configuration**:
```python
IMAGE_RESOLUTION_SCALE = 2.0  # Higher = better quality, larger files
```

---

## Common Prerequisites

### Core Dependencies

All examples require:
```bash
pip install docling
```

### Optional Dependencies

Depending on the example:

**For VLM examples**:
- Ollama: https://ollama.ai
- Models: `ollama pull gabegoodhart/granite-docling:258M` or `ollama pull ibm/granite3.3-vision:2b`

**For GPU examples**:
- NVIDIA GPU with CUDA support
- CUDA toolkit
- cuDNN (if required)

**For OCR examples**:
- Tesseract: https://github.com/tesseract-ocr/tesseract
- RapidOCR: `pip install rapidocr-onnxruntime`
- EasyOCR: `pip install easyocr`

**For image processing**:
- PIL/Pillow: `pip install Pillow` (usually included with docling)

---

## Tips & Best Practices

### 1. Start Simple
Begin with `basic/simple_processor.py` to understand the fundamentals before moving to advanced examples.

### 2. Choose the Right Processing Method

- **Default processing**: Fast, good for most documents with embedded text
- **GPU processing**: Use for batch processing or large documents
- **VLM processing**: Best for complex layouts, formulas, and visual understanding
- **OCR processing**: Required for scanned documents or image-based PDFs

### 3. VLM Models Comparison

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| gabegoodhart/granite-docling:258M | 258MB | Document OCR, layout | Fast |
| ibm/granite3.3-vision:2b | 2B | Image understanding, annotations | Medium |

### 4. OCR Engine Selection

| Engine | Speed | Accuracy | Dependencies | Languages |
|--------|-------|----------|--------------|-----------|
| RapidOCR | Fast | Good | None | Limited |
| Tesseract | Medium | Excellent | System | 100+ |
| EasyOCR | Slow | Excellent | GPU helpful | 80+ |
| OcrMac | Fast | Good | macOS only | Auto |

### 5. Performance Optimization

- Use GPU acceleration for batch processing
- Adjust image scale based on quality needs (1.0 = 72 DPI, 2.0 = 144 DPI)
- For large documents, consider processing pages in batches
- Use VLM for complex documents, OCR for simpler scanned docs

### 6. Error Handling

All examples include error handling for common issues:
- File not found
- Network connectivity (for URLs)
- Missing dependencies (Ollama, Tesseract, etc.)
- GPU not available

Check error messages for installation instructions.

### 7. Environment Variables

Several examples support environment variables for configuration:

```bash
# Ollama configuration
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="gabegoodhart/granite-docling:258M"
export VISION_MODEL="ibm/granite3.3-vision:2b"
```

### 8. Output Formats

Docling supports multiple export formats:

| Format | Use Case | Image Handling |
|--------|----------|----------------|
| Markdown | General text processing | Embedded or referenced |
| HTML | Web display | Referenced images |
| JSON | Programmatic access | References |
| Docling JSON | Full document structure | Complete metadata |

---

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Start Ollama
ollama serve
```

### Tesseract Not Found
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-fra

# macOS
brew install tesseract tesseract-lang

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA toolkit for your platform
```

### Import Errors
```bash
# Install all optional dependencies
pip install docling[all]

# Or install specific components
pip install rapidocr-onnxruntime  # For RapidOCR
pip install easyocr  # For EasyOCR
```

---

## Next Steps

After exploring these examples:

1. **Modify for your use case**: Adapt examples to your specific documents and requirements
2. **Combine techniques**: Mix VLM, OCR, and GPU acceleration as needed
3. **Build pipelines**: Use `figure_export.py` as a template for comprehensive processing
4. **Contribute**: Share your improvements or new examples with the community

---

## Additional Resources

- **Docling Documentation**: https://github.com/DS4SD/docling
- **Ollama Documentation**: https://ollama.ai/docs
- **Tesseract Documentation**: https://github.com/tesseract-ocr/tesseract
- **CorpusCraft Main README**: [../README.md](../README.md)

---

## Questions or Issues?

- Check the main project README: [../README.md](../README.md)
- Review Docling documentation
- Open an issue in the repository
