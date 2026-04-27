from pathlib import Path

from corpuscraft.config import ParserConfig, PipelineType
from corpuscraft.parsers.factory import create_parser
from corpuscraft.preprocessing.poppler import PopplerPreprocessor
from corpuscraft.routing import PipelineRouter


def main() -> None:
    pdf_path = Path("data/raw/bulletin-de-paie-du-011025-au-311025.pdf")

    # Step 1 — preprocess
    pre = PopplerPreprocessor(clean=True, split=False, rasterize=False)
    preprocessed = pre.run(pdf_path, Path("outputs/preprocessed") / pdf_path.stem)

    print(f"Preprocessed : {preprocessed}")
    print(f"Scanned      : {preprocessed.is_scanned}")

    # Step 2 — auto-route: inspects content and picks the best pipeline
    router = PipelineRouter(detection_level="basic")
    result = router.route(preprocessed)
    print(f"Pipeline     : {result.pipeline.value}")
    print(f"Reason       : {result.reason}")
    print(f"Confidence   : {result.confidence:.0%}")
    if result.alternatives:
        print(f"Alternatives : {', '.join(p.value for p in result.alternatives)}")
    pipeline = result.pipeline

    parser = create_parser(ParserConfig(pipeline=pipeline))

    # Override options — pick any one:
    # doc = parser.parse_file(preprocessed.source_path)       # original, unmodified PDF
    # doc = parser.parse_file(preprocessed.cleaned_pdf)       # cleaned PDF (invisible text + annotations stripped)
    # doc = parser.parse_file(preprocessed.page_pdfs[0])      # single page from split (requires split=True)
    # parser = create_parser(ParserConfig(pipeline=PipelineType.standard))  # force a specific parser
    doc = parser.parse_file(preprocessed.parser_input())

    print(f"\nParsed       : {doc}")
    print(f"Characters   : {len(doc):,}")
    print(f"\nContent preview:\n{'─' * 40}")
    print(doc.content[:600])
    print("─" * 40)

    out = Path("outputs/parsed") / (pdf_path.stem + "_preprocessed.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc.content, encoding="utf-8")
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
