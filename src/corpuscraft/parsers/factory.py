from __future__ import annotations

from corpuscraft.config import ParserConfig, PipelineType
from corpuscraft.parsers.base import BaseParser


def create_parser(config: ParserConfig) -> BaseParser:
    match config.pipeline:
        case PipelineType.vlm:
            from corpuscraft.parsers.vlm import VlmParser
            return VlmParser(config)
        case PipelineType.ocr:
            from corpuscraft.parsers.ocr import OcrParser
            return OcrParser(config)
        case PipelineType.gpu:
            from corpuscraft.parsers.standard import StandardParser
            return StandardParser(config, use_gpu=True)
        case PipelineType.yolo:
            from corpuscraft.parsers.yolo import YoloParser
            return YoloParser(config)
        case PipelineType.mineru:
            from corpuscraft.parsers.mineru import MineruParser
            return MineruParser(config)
        case PipelineType.consensus:
            from corpuscraft.parsers.consensus import ConsensusParser
            return ConsensusParser(config)
        case PipelineType.pymupdf:
            from corpuscraft.parsers.pymupdf_parser import PyMuPDFParser
            return PyMuPDFParser(config)
        case PipelineType.pdfplumber:
            from corpuscraft.parsers.pdfplumber_parser import PdfPlumberParser
            return PdfPlumberParser(config)
        case PipelineType.python_docx:
            from corpuscraft.parsers.python_docx_parser import PythonDocxParser
            return PythonDocxParser(config)
        case PipelineType.python_pptx:
            from corpuscraft.parsers.python_pptx_parser import PythonPptxParser
            return PythonPptxParser(config)
        case PipelineType.mammoth:
            from corpuscraft.parsers.mammoth_parser import MammothParser
            return MammothParser(config)
        case PipelineType.markitdown:
            from corpuscraft.parsers.markitdown_parser import MarkItDownParser
            return MarkItDownParser(config)
        case _:
            from corpuscraft.parsers.standard import StandardParser
            return StandardParser(config)
