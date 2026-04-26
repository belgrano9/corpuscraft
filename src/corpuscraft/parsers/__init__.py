"""
Document parsers for CorpusCraft.
"""

from corpuscraft.parsers.docling_parser import DoclingParser

__all__ = ["DoclingParser"]

try:
    from corpuscraft.parsers.yolo import YoloParser

    __all__ = [*__all__, "YoloParser"]
except ImportError:
    pass
