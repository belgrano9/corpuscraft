"""
Example usage patterns for the CorpusCraft Docling parser.

This file demonstrates various ways to use the DoclingParser in your code.
"""

from pathlib import Path

from corpuscraft.parsers import DoclingParser

# Example 1: Using Ollama VLM backend (DEFAULT)
print("\n" + "=" * 60)
print("Example 1: Ollama VLM Backend (Default - Enhanced OCR)")
print("=" * 60)

# Ollama is now enabled by default!
# You can customize the model and settings
parser_ollama = DoclingParser(
    # use_ollama=True is the default now
    ollama_model="gabegoodhart/granite-docling:258M",  # Specialized OCR model
    ocr_engine="tesseract",  # or "easyocr" for handwritten text
    vlm_scale=2.0,  # Higher resolution for better math formula recognition
)

# Parse a scanned PDF or image with mathematical content
doc = parser_ollama.parse_file("data/2408.09869v5.pdf")
if doc:
    print(f"Parsed with Ollama VLM: {doc.file_path}")
    print("This will preserve LaTeX formulas and complex layouts")


# Example 2: Custom VLM prompt for specific extraction needs
print("\n" + "=" * 60)
print("Example 2: Custom VLM Prompt")
print("=" * 60)

custom_prompt = """
Extract all text from this document, focusing on:
- Code snippets (preserve syntax and indentation)
- Technical diagrams (describe them)
- Tables (preserve structure)
- Equations (use LaTeX notation)
"""

parser_custom = DoclingParser(
    custom_prompt=custom_prompt,  # Ollama is already default
)

doc = parser_custom.parse_file("data/2408.09869v5.pdf")
if doc:
    print(f"Parsed with custom prompt: {doc.file_path}")


# Example 5: Accessing metadata
print("\n" + "=" * 60)
print("Example 5: Accessing Document Metadata")
print("=" * 60)

parser = DoclingParser()
doc = parser.parse_file("data/example.pdf")

if doc:
    print(f"File: {doc.file_path}")
    print(f"Format: {doc.file_format}")
    print(f"Size: {doc.metadata.get('file_size', 0):,} bytes")
    print(f"Pages: {doc.page_count}")
    print(f"Character count: {len(doc)}")


# Example 6: Working with image files (requires Ollama VLM)
print("\n" + "=" * 60)
print("Example 6: Parsing Images with OCR")
print("=" * 60)

parser_vlm = DoclingParser(use_ollama=True)

# Parse screenshots, scanned documents, photos of text
for image_file in Path("data").glob("*.png"):
    doc = parser_vlm.parse_file(image_file)
    if doc:
        print(f"Extracted text from {image_file.name}")
        print(f"  Content: {doc.content[:100]}...")


print("\n" + "=" * 60)
print("Examples completed!")
print("=" * 60)
