"""
Minimal MinerU parsing test.

Parses a PDF with MinerU's pipeline backend (PPDocLayoutV2 layout detection,
UnimerNet formula recognition, PaddleOCR, table reconstruction) and saves
the resulting markdown to scratch/.

Usage:
    uv run examples/advanced/simple_mineru.py
    uv run examples/advanced/simple_mineru.py path/to/file.pdf
"""

import sys
from pathlib import Path


def main() -> None:
    from corpuscraft.config import ParserConfig, PipelineType
    from corpuscraft.parsers.factory import create_parser

    source_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/2408.09869v5.pdf")
    output_dir = Path("scratch/mineru")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (source_file.stem + ".md")

    print(f"Input:  {source_file}")
    print(f"Output: {output_file}")

    parser = create_parser(ParserConfig(pipeline=PipelineType.mineru))
    doc = parser.parse_file(source_file)

    output_file.write_text(doc.content, encoding="utf-8")

    words = len(doc.content.split())
    headings = doc.content.count("\n#")
    tables = doc.content.count("\n|")

    print(f"\nParsed  {len(doc.content):,} chars  |  ~{words:,} words  |  {headings} headings  |  {tables} table rows")
    print(f"\n--- first 1000 chars ---\n{doc.content[:1000]}")


if __name__ == "__main__":
    main()
