import logging
from pathlib import Path

from corpuscraft.config import ParserConfig, PipelineType
from corpuscraft.parsers.factory import create_parser

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    # 1. Define the input file and output path
    pdf_path = Path("data/raw/bulletin-de-paie-du-011025-au-311025.pdf")
    output_md_path = Path("outputs/pymupdf_example_output.md")
    
    # Make sure output directory exists
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing document: {pdf_path}")
    print("This will use the pure PyMuPDF (via pymupdf4llm) parser...")

    # 2. Configure the parser
    # We specify PipelineType.pymupdf to use the PyMuPDFParser
    config = ParserConfig(pipeline=PipelineType.pymupdf)

    # 3. Create the parser using the factory
    parser = create_parser(config)

    # 4. Parse the file
    parsed_doc = parser.parse_file(pdf_path)

    # 5. Output results
    print("\n--- Parsing Complete ---")
    print(f"Pipeline used: {parsed_doc.pipeline}")
    print(f"Characters extracted: {len(parsed_doc)}")
    
    print(f"\nMetadata extracted:")
    for key, value in parsed_doc.metadata.items():
        print(f"  {key}: {value}")
        
    print(f"\nPreview of markdown content (first 500 chars):")
    print("-" * 40)
    print(parsed_doc.content[:500] + "\n...")
    print("-" * 40)

    # Save to a markdown file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(parsed_doc.content)
        
    print(f"\nFull markdown saved to: {output_md_path}")


if __name__ == "__main__":
    main()
