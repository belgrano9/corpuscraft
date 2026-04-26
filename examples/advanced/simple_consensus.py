"""
Consensus parser example.

Runs Docling, YOLO, and MinerU on the same PDF, validates tables and figures
by bounding-box agreement (IoM), and saves the resulting markdown to scratch/.

Body text comes from MinerU. Tables and figures are only included when at
least one other parser also detects them in the same region.

Usage:
    uv run examples/advanced/simple_consensus.py
    uv run examples/advanced/simple_consensus.py path/to/file.pdf
"""

import sys
from pathlib import Path


def main() -> None:
    from corpuscraft.config import ParserConfig, PipelineType
    from corpuscraft.parsers.factory import create_parser

    source_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/2408.09869v5.pdf")
    output_dir = Path("scratch/consensus")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (source_file.stem + ".md")

    print(f"Input:  {source_file}")
    print(f"Output: {output_file}")

    parser = create_parser(ParserConfig(pipeline=PipelineType.consensus))
    doc = parser.parse_file(source_file)

    output_file.write_text(doc.content, encoding="utf-8")

    words = len(doc.content.split())
    headings = doc.content.count("\n#")
    tables = sum(1 for line in doc.content.splitlines() if line.startswith("|") and "---" in line)

    print(f"\nParsed  {len(doc.content):,} chars  |  ~{words:,} words  |  {headings} headings  |  {tables} confirmed tables")
    print(f"\n--- first 1000 chars ---\n{doc.content[:1000]}")


if __name__ == "__main__":
    main()
