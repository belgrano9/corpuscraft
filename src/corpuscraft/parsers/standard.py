from __future__ import annotations

from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser


class StandardParser(BaseParser):
    def __init__(self, config: ParserConfig, use_gpu: bool = False) -> None:
        self._pipeline = "gpu" if use_gpu else "standard"
        if use_gpu:
            pipeline_options = ThreadedPdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
            )
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=ThreadedStandardPdfPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
        else:
            self._converter = DocumentConverter()

    def parse_file(self, path: Path) -> ParsedDocument:
        result = self._converter.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        metadata = {
            "file_size": path.stat().st_size,
            "num_pages": len(doc.pages) if hasattr(doc, "pages") else None,
        }
        return ParsedDocument(
            content=content,
            source_path=path,
            pipeline=self._pipeline,
            metadata=metadata,
        )
