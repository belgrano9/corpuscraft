from __future__ import annotations

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser


# EasyOCR uses 2-letter codes; Tesseract uses 3-letter codes.
_TESSERACT_TO_EASYOCR: dict[str, str] = {
    "eng": "en", "fra": "fr", "deu": "de", "spa": "es",
    "ita": "it", "por": "pt", "rus": "ru", "ara": "ar",
    "jpn": "ja", "kor": "ko", "chi": "ch_sim",
}


def _to_easyocr_langs(languages: list[str]) -> list[str]:
    return [_TESSERACT_TO_EASYOCR.get(lang, lang) for lang in languages]


def _build_ocr_options(engine: str, languages: list[str]) -> object:
    if engine == "tesseract":
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        return TesseractCliOcrOptions(force_full_page_ocr=True, lang=languages)
    if engine == "easyocr":
        from docling.datamodel.pipeline_options import EasyOcrOptions
        return EasyOcrOptions(force_full_page_ocr=True, lang=_to_easyocr_langs(languages))
    # default: rapidocr
    from docling.datamodel.pipeline_options import RapidOcrOptions
    return RapidOcrOptions(force_full_page_ocr=True)


class OcrParser(BaseParser):
    def __init__(self, config: ParserConfig) -> None:
        self._config = config
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options = _build_ocr_options(
            config.ocr_engine, config.ocr_languages
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_file(self, path: Path) -> ParsedDocument:
        result = self._converter.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        metadata = {
            "file_size": path.stat().st_size,
            "ocr_engine": self._config.ocr_engine,
            "ocr_languages": self._config.ocr_languages,
            "num_pages": len(doc.pages) if hasattr(doc, "pages") else None,
        }
        return ParsedDocument(
            content=content,
            source_path=path,
            pipeline="ocr",
            metadata=metadata,
        )
