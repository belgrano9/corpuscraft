from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode

def main() -> None:
    # 1. Path to the PPTX file
    pptx_path = Path("data/raw/Results3.pptx")
    
    if not pptx_path.exists():
        print(f"Error: Could not find {pptx_path}")
        return

    print("Parsing PPTX using Docling directly to extract embedded images...")

    # 2. Parse the file directly using Docling
    converter = DocumentConverter()
    result = converter.convert(str(pptx_path))
    doc = result.document

    # 3. Export to Markdown WITH embedded images
    # By default, doc.export_to_markdown() uses ImageRefMode.PLACEHOLDER
    # which results in '<!-- image -->' tags.
    # We change it to EMBEDDED to get base64 data URIs.
    md_content = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)

    # 4. Display results
    print(f"\nExtracted {len(doc.pages)} slides")
    print(f"Characters in markdown: {len(md_content):,}")
    
    print(f"\nContent preview:\n{'-' * 40}")
    print(md_content[:600] + "\n..." if len(md_content) > 600 else md_content)
    print("-" * 40)

    # 5. Export to Markdown
    out = Path("outputs/parsed") / (pptx_path.stem + "_with_images.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved full markdown with embedded images to: {out}")

if __name__ == "__main__":
    main()
