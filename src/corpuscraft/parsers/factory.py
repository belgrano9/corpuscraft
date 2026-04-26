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
        case _:
            from corpuscraft.parsers.standard import StandardParser
            return StandardParser(config)
