from __future__ import annotations

import urllib.request
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

_PROMPT = (
    "Convert this page to docling format. "
    "Extract all text, preserving the exact layout and structure. "
    "Be precise and do not skip any content."
)


def check_ollama_connection(host: str, timeout: int = 5) -> bool:
    try:
        urllib.request.urlopen(host, timeout=timeout)
        return True
    except Exception:
        return False


class VlmParser(BaseParser):
    def __init__(self, config: ParserConfig) -> None:
        self._config = config
        endpoint = f"{config.vlm_host}/v1/chat/completions"
        pipeline_options = VlmPipelineOptions(enable_remote_services=True)
        pipeline_options.vlm_options = ApiVlmOptions(
            url=endpoint,
            params={"model": config.vlm_model},
            prompt=_PROMPT,
            timeout=300,
            scale=config.image_scale,
            response_format=ResponseFormat.DOCTAGS,
        )
        format_opt = PdfFormatOption(
            pipeline_options=pipeline_options,
            pipeline_cls=VlmPipeline,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: format_opt,
                InputFormat.IMAGE: format_opt,
            }
        )

    def parse_file(self, path: Path) -> ParsedDocument:
        if not check_ollama_connection(self._config.vlm_host):
            raise RuntimeError(
                f"Cannot reach Ollama at {self._config.vlm_host}. "
                "Make sure Ollama is running."
            )
        result = self._converter.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        metadata = {
            "file_size": path.stat().st_size,
            "vlm_model": self._config.vlm_model,
            "num_pages": len(doc.pages) if hasattr(doc, "pages") else None,
        }
        return ParsedDocument(
            content=content,
            source_path=path,
            pipeline="vlm",
            metadata=metadata,
        )
