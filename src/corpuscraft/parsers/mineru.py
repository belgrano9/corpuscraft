from __future__ import annotations

from pathlib import Path

from loguru import logger

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser


class _NullWriter:
    """Discards images extracted by MinerU — we only need text."""

    def write(self, path: str, data: bytes) -> None:
        pass


class MineruParser(BaseParser):
    """PDF parser backed by MinerU's pipeline backend.

    Calls MinerU's internal Python API directly (no subprocess).
    The on_doc_ready callback intercepts the intermediate 'middle JSON'
    (structured blocks with bboxes + content) before markdown conversion,
    making it straightforward to customise extraction in the future.

    Install the extra: uv pip install "mineru[pipeline]"
    Models are downloaded automatically on first use to ~/.mineru/.
    """

    def __init__(self, config: ParserConfig) -> None:
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        try:
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown
            from mineru.utils.enum_class import MakeMode
        except ImportError as e:
            raise ImportError(
                "MinerU is not installed. Run: uv pip install 'mineru[pipeline]'"
            ) from e

        pdf_bytes = path.read_bytes()
        collected: list[str] = []

        def on_doc_ready(
            doc_index: int,
            model_list: list,
            middle_json: dict,
            ocr_enable: bool,
        ) -> None:
            for page in middle_json.get("pdf_info", []):
                para_blocks = page.get("para_blocks", [])
                lines = make_blocks_to_markdown(para_blocks, MakeMode.NLP_MD)
                collected.extend(lines)

        logger.info(f"MinerU parsing {path.name}")
        doc_analyze_streaming(
            pdf_bytes_list=[pdf_bytes],
            image_writer_list=[_NullWriter()],
            lang_list=[""],
            on_doc_ready=on_doc_ready,
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
        )

        content = "\n\n".join(line for line in collected if line.strip())
        metadata: dict = {"file_size": path.stat().st_size}

        return ParsedDocument(
            content=content,
            source_path=path,
            pipeline="mineru",
            metadata=metadata,
        )
