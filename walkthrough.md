# CorpusCraft: A Linear Code Walkthrough

*2026-04-27T12:11:26Z by Showboat 0.6.1*
<!-- showboat-id: 77012b69-be02-45ca-a5cb-0b88384fd8b1 -->

CorpusCraft converts PDF documents into training datasets for LLMs. It chains five stages together: preprocess → detect → route → parse → generate → export. Each stage is independently testable and the entire chain is driven by plain Python API calls — no CLI.

This walkthrough traces the code in execution order, from config to final JSONL output.

```bash
find src/corpuscraft -name '*.py' | sort
```

```output
src/corpuscraft/__init__.py
src/corpuscraft/config.py
src/corpuscraft/exporters/__init__.py
src/corpuscraft/exporters/jsonl.py
src/corpuscraft/generators/__init__.py
src/corpuscraft/generators/base.py
src/corpuscraft/generators/qa.py
src/corpuscraft/models.py
src/corpuscraft/parsers/__init__.py
src/corpuscraft/parsers/base.py
src/corpuscraft/parsers/consensus.py
src/corpuscraft/parsers/factory.py
src/corpuscraft/parsers/mineru.py
src/corpuscraft/parsers/ocr.py
src/corpuscraft/parsers/pdfplumber_parser.py
src/corpuscraft/parsers/pymupdf_parser.py
src/corpuscraft/parsers/standard.py
src/corpuscraft/parsers/vlm.py
src/corpuscraft/parsers/yolo.py
src/corpuscraft/preprocessing/__init__.py
src/corpuscraft/preprocessing/base.py
src/corpuscraft/preprocessing/models.py
src/corpuscraft/preprocessing/poppler.py
src/corpuscraft/routing/__init__.py
src/corpuscraft/routing/detector.py
src/corpuscraft/routing/models.py
src/corpuscraft/routing/router.py
src/corpuscraft/routing/rules.py
```

## Stage 1: Configuration

Everything starts in `config.py`. It defines a `PipelineType` enum naming every parsing backend, and a hierarchy of Pydantic models that describe the full job: preprocessing options, which parser to use, LLM settings, how many examples to generate, and where to write them.

The top-level class is `CorpusCraftConfig`. You can build one in code or load it from a YAML file via `load_config(path)`.

```bash
grep -n 'class PipelineType\|    [a-z]' src/corpuscraft/config.py | head -20
```

```output
10:class PipelineType(str, Enum):
11:    standard = "standard"
12:    gpu = "gpu"
13:    vlm = "vlm"
14:    ocr = "ocr"
15:    yolo = "yolo"
16:    mineru = "mineru"
17:    consensus = "consensus"
18:    pymupdf = "pymupdf"
19:    pdfplumber = "pdfplumber"
23:    clean: bool = True
24:    split: bool = False
25:    rasterize: bool = False
26:    raster_dpi: int = 150
27:    raster_format: str = "png"
28:    scanned_text_threshold: int = 100
32:    pipeline: PipelineType = PipelineType.standard
33:    ocr_engine: str = "rapidocr"
34:    ocr_languages: list[str] = Field(default_factory=lambda: ["eng"])
35:    vlm_model: str = "gabegoodhart/granite-docling:258M"
```

```bash
grep -n '^class ' src/corpuscraft/config.py
```

```output
10:class PipelineType(str, Enum):
22:class PreprocessingConfig(BaseModel):
31:class ParserConfig(BaseModel):
44:class LLMConfig(BaseModel):
51:class GeneratorConfig(BaseModel):
56:class ExporterConfig(BaseModel):
62:class CorpusCraftConfig(BaseModel):
```

```bash
sed -n '62,80p' src/corpuscraft/config.py
```

```output
class CorpusCraftConfig(BaseModel):
    input_dir: Path
    parser: ParserConfig = Field(default_factory=ParserConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generators: list[GeneratorConfig] = Field(default_factory=list)
    exporter: ExporterConfig = Field(default_factory=ExporterConfig)


def load_config(path: Path) -> CorpusCraftConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return CorpusCraftConfig.model_validate(data)


def save_default_config(path: Path, input_dir: Path, output_dir: Path) -> None:
    cfg = CorpusCraftConfig(
        input_dir=input_dir,
        generators=[GeneratorConfig(type="qa")],
        exporter=ExporterConfig(output_dir=output_dir),
```

Notice that `CorpusCraftConfig` has no `preprocessing` field — preprocessing is configured separately when you instantiate `PopplerPreprocessor`. The config tree only covers parser, LLM, generator, and exporter settings.

## Stage 2: Core Data Models

`models.py` defines the two data shapes that flow through the system. `ParsedDocument` is what every parser produces. `QAExample` is what every generator produces. Everything downstream — generators, exporters — depends only on these two types.

```bash
cat src/corpuscraft/models.py
```

```output
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    content: str
    source_path: Path
    pipeline: str
    metadata: dict = field(default_factory=dict)
    chunks: list[str] | None = None

    def __len__(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        return (
            f"ParsedDocument(source={self.source_path.name!r}, "
            f"pipeline={self.pipeline!r}, chars={len(self)})"
        )


@dataclass
class QAExample:
    question: str
    answer: str
    context: str
    source: str
    difficulty: str = "medium"
```

## Stage 3: Preprocessing (poppler-utils)

Before any parser runs, `PopplerPreprocessor` cleans the PDF and gathers facts the router needs. It wraps five poppler command-line tools in sequence:

1. **inspect** — `pdfinfo` → page count, encryption flag, dimensions
2. **probe** — `pdftotext` on pages 1-3 → counts extracted characters; fewer than 100 means scanned
3. **clean** — `pdftocairo -pdf` → strips annotations, JavaScript, form fields, invisible text
4. **split** — `pdfseparate` → one PDF per page (optional)
5. **rasterize** — `pdftoppm` → PNG/JPEG page images at configurable DPI (optional)

The result is a `PreprocessedPDF` dataclass. Its `parser_input()` method returns the cleaned PDF if cleaning ran, or the original otherwise — parsers always call this instead of touching the source directly.

```bash
cat src/corpuscraft/preprocessing/models.py
```

```output
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PDFMetadata:
    page_count: int
    encrypted: bool
    pdf_version: str
    title: str
    author: str
    page_size: str
    raw: dict = field(default_factory=dict)


@dataclass
class PreprocessedPDF:
    source_path: Path
    output_dir: Path
    metadata: PDFMetadata
    is_scanned: bool
    cleaned_pdf: Path | None = None
    page_pdfs: list[Path] = field(default_factory=list)
    page_images: list[Path] = field(default_factory=list)

    def parser_input(self) -> Path:
        """Best path to feed into a parser: cleaned PDF if available, else original."""
        return self.cleaned_pdf if self.cleaned_pdf is not None else self.source_path

    def __repr__(self) -> str:
        return (
            f"PreprocessedPDF(source={self.source_path.name!r}, "
            f"pages={self.metadata.page_count}, "
            f"scanned={self.is_scanned}, "
            f"cleaned={self.cleaned_pdf is not None})"
        )
```

```bash
grep -n 'def \|class \|binary_path\|scanned_text_threshold' src/corpuscraft/preprocessing/poppler.py | head -35
```

```output
11:def _require_binary(name: str) -> str:
22:class PopplerPreprocessor(BasePreprocessor):
35:    def __init__(
42:        scanned_text_threshold: int = 100,
49:        self.scanned_text_threshold = scanned_text_threshold
58:    def run(self, pdf_path: Path, output_dir: Path) -> PreprocessedPDF:
82:    def _inspect(self, pdf_path: Path) -> PDFMetadata:
105:    def _probe_text(self, pdf_path: Path) -> bool:
113:        return len(result.stdout.strip()) < self.scanned_text_threshold
115:    def _clean(self, pdf_path: Path, output_dir: Path) -> Path:
124:    def _split(self, pdf_path: Path, output_dir: Path) -> list[Path]:
135:    def _rasterize(self, pdf_path: Path, output_dir: Path) -> list[Path]:
```

```bash
sed -n '58,81p' src/corpuscraft/preprocessing/poppler.py
```

```output
    def run(self, pdf_path: Path, output_dir: Path) -> PreprocessedPDF:
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._inspect(pdf_path)
        is_scanned = self._probe_text(pdf_path)

        cleaned_pdf = self._clean(pdf_path, output_dir) if self.clean else None
        page_pdfs = self._split(pdf_path, output_dir) if self.split else []
        page_images = self._rasterize(pdf_path, output_dir) if self.rasterize else []

        return PreprocessedPDF(
            source_path=pdf_path,
            output_dir=output_dir,
            metadata=metadata,
            is_scanned=is_scanned,
            cleaned_pdf=cleaned_pdf,
            page_pdfs=page_pdfs,
            page_images=page_images,
        )

    # ------------------------------------------------------------------
    # Private operations
    # ------------------------------------------------------------------

```

## Stage 4: Routing

The router's job is to pick the right parser without you having to think about it. It has three sub-components that compose together:

- **`ContentDetector`** — inspects the PDF (with fitz or YOLO) and produces a `ContentProfile`
- **`RoutingRules`** — pure function, no I/O; takes a `ContentProfile` and returns a `RoutingResult`
- **`PipelineRouter`** — facade that wires them together; exposes `route(preprocessed)` and `route_path(path)`

Detection depth is controlled by the `detection_level` argument: `none` (metadata only), `basic` (fitz analysis, fast), or `enhanced` (basic + DocLayout-YOLO, slower but more accurate).

```bash
cat src/corpuscraft/routing/models.py
```

```output
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from corpuscraft.config import PipelineType


@dataclass
class ContentProfile:
    # Tier 0 — always populated from PreprocessedPDF (no detection needed)
    extension: str          # lowercase with dot: ".pdf", ".docx", etc.
    page_count: int
    is_scanned: bool
    encrypted: bool
    page_size: str          # raw pdfinfo string, e.g. "595 x 842 pts"

    # Tier 1 — "basic" detection via pymupdf/fitz (None = not measured)
    image_ratio: float | None = None          # image block area / page area, 0.0–1.0
    table_count: int | None = None            # tables found via page.find_tables()
    has_formula_heuristic: bool | None = None # LaTeX regex + unicode math chars
    is_multi_column: bool | None = None       # bimodal X-position histogram test
    text_density: float | None = None         # avg chars/page across sampled pages

    # Tier 2 — "enhanced" detection via YOLO (None = YOLO was not run)
    yolo_label_counts: dict[str, int] | None = None  # e.g. {"Table": 3, "Formula": 1}
    yolo_avg_confidence: float | None = None

    detection_level: Literal["none", "basic", "enhanced"] = "none"


@dataclass
class RoutingResult:
    pipeline: PipelineType
    reason: str             # human-readable, safe to log
    confidence: float       # 0.0 (pure guess) – 1.0 (certain)
    alternatives: list[PipelineType] = field(default_factory=list)
    profile: ContentProfile | None = None
```

The `ContentProfile` tiers directly map to the three detection levels. Tier-0 fields are always set (they come from poppler metadata, which is free). Tier-1 fields require opening the PDF with fitz. Tier-2 fields require running YOLO on rendered page images. Fields left `None` were never measured — the routing rules treat `None` as 'signal absent', not as zero.

```bash
grep -n 'def \|class \|_NON_PDF\|TABLE_THRESHOLD\|FORMULA_WEIGHT\|score' src/corpuscraft/routing/rules.py | head -40
```

```output
9:_NON_PDF_ROUTES: dict[str, PipelineType] = {
29:_SCORE_MINERU: int = 2                   # score 2–3 → mineru
30:_SCORE_CONSENSUS: int = 4               # score ≥4 → consensus
35:class RoutingRules:
43:    def route(profile: ContentProfile) -> RoutingResult:
63:    def _route_non_pdf(profile: ContentProfile) -> RoutingResult | None:
64:        pipeline = _NON_PDF_ROUTES.get(profile.extension)
81:    def _route_encrypted(profile: ContentProfile) -> RoutingResult | None:
97:    def _route_scanned(profile: ContentProfile) -> RoutingResult | None:
136:    def _complexity_score(profile: ContentProfile) -> tuple[int, list[str]]:
137:        score = 0
147:            score += 2
157:            score += 1
162:            score += 1
172:            score += 1
177:            score += 1
180:        return score, reasons
183:    def _route_native_pdf(profile: ContentProfile) -> RoutingResult:
184:        score, detected = RoutingRules._complexity_score(profile)
196:        if score == 0:
205:        if score >= _SCORE_CONSENSUS:
208:                reason=f"High-complexity native PDF (score={score}, features: {features}): consensus validates structural elements across multiple parsers",
214:        if score >= _SCORE_MINERU:
217:                reason=f"Medium-complexity native PDF (score={score}, features: {features}): mineru handles formulas, multi-column, and tables",
223:        # score == 1: single-feature specialist
227:                reason=f"Native PDF with formulas (score=1): mineru is the only non-VLM parser with formula support",
236:                reason=f"Native PDF with tables only (score=1): pdfplumber has reliable table extraction at low cost",
245:                reason=f"Native PDF with multi-column layout (score=1): yolo handles layout-aware reading order",
254:                reason=f"Native PDF with significant figure content (score=1): yolo detects and labels figure regions",
273:            reason=f"Native PDF, unhandled score branch (score={score}, features: {features}): fallback to pymupdf",
```

```bash
sed -n '43,62p' src/corpuscraft/routing/rules.py
```

```output
    def route(profile: ContentProfile) -> RoutingResult:
        result = RoutingRules._route_non_pdf(profile)
        if result is not None:
            return result

        result = RoutingRules._route_encrypted(profile)
        if result is not None:
            return result

        result = RoutingRules._route_scanned(profile)
        if result is not None:
            return result

        return RoutingRules._route_native_pdf(profile)

    # ------------------------------------------------------------------
    # Gate 1: non-PDF extensions
    # ------------------------------------------------------------------

    @staticmethod
```

```bash
sed -n '136,182p' src/corpuscraft/routing/rules.py
```

```output
    def _complexity_score(profile: ContentProfile) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []

        # Formulas (weight 2) — YOLO preferred; heuristic fallback
        formula_detected = False
        if profile.yolo_label_counts is not None:
            formula_detected = profile.yolo_label_counts.get("Formula", 0) > 0
        elif profile.has_formula_heuristic is True:
            formula_detected = True
        if formula_detected:
            score += 2
            reasons.append("formulas")

        # Tables (weight 1) — YOLO preferred; fitz fallback
        table_detected = False
        if profile.yolo_label_counts is not None:
            table_detected = profile.yolo_label_counts.get("Table", 0) > 0
        elif profile.table_count is not None:
            table_detected = profile.table_count > 0
        if table_detected:
            score += 1
            reasons.append("tables")

        # Multi-column layout (weight 1)
        if profile.is_multi_column is True:
            score += 1
            reasons.append("multi-column")

        # Figure-heavy (weight 1) — YOLO preferred; image_ratio fallback
        figure_heavy = False
        if profile.yolo_label_counts is not None:
            figure_heavy = profile.yolo_label_counts.get("Picture", 0) >= _MIN_YOLO_FIGURES_HEAVY
        elif profile.image_ratio is not None:
            figure_heavy = profile.image_ratio > _IMAGE_RATIO_FIGURE_HEAVY
        if figure_heavy:
            score += 1
            reasons.append("figure-heavy" if profile.yolo_label_counts is not None else "image-heavy")

        # Low text density on a native PDF (weight 1)
        if profile.text_density is not None and profile.text_density < _TEXT_DENSITY_LOW:
            score += 1
            reasons.append("low-text-density")

        return score, reasons

    @staticmethod
```

The complexity score is intentionally additive and weighted: formulas count double because they require specialized LaTeX-aware parsing that only MinerU and Consensus provide. Each feature flag flips independently — YOLO signals take priority over fitz heuristics when available, so the same property (e.g. 'has a table') is measured at the best available precision.

```bash
cat src/corpuscraft/routing/router.py
```

```output
from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger

from corpuscraft.preprocessing.models import PDFMetadata, PreprocessedPDF
from corpuscraft.routing.detector import ContentDetector
from corpuscraft.routing.models import ContentProfile, RoutingResult
from corpuscraft.routing.rules import RoutingRules


class PipelineRouter:
    """
    Facade that combines ContentDetector and RoutingRules into a single call.

    Typical usage::

        router = PipelineRouter(detection_level="basic")

        # After running PopplerPreprocessor (recommended — accurate is_scanned):
        result = router.route(preprocessed)

        # From a bare file path (is_scanned defaults to False):
        result = router.route_path(Path("paper.pdf"))

        print(result.pipeline.value)   # e.g. "mineru"
        print(result.reason)           # human-readable explanation
        print(result.confidence)       # 0.0 – 1.0
        print(result.alternatives)     # ranked fallback pipelines
    """

    def __init__(
        self,
        detection_level: Literal["none", "basic", "enhanced"] = "basic",
        sample_pages: int = 5,
        yolo_model_id: str = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
        yolo_confidence: float = 0.2,
    ) -> None:
        self._detector = ContentDetector(
            level=detection_level,
            sample_pages=sample_pages,
            yolo_model_id=yolo_model_id,
            yolo_confidence=yolo_confidence,
        )

    def route(self, preprocessed: PreprocessedPDF) -> RoutingResult:
        """
        Primary routing path. Requires a PreprocessedPDF from PopplerPreprocessor
        so that is_scanned and metadata are accurate.
        """
        profile = self._detector.detect(preprocessed)
        result = RoutingRules.route(profile)
        logger.debug(
            "route({name}): {pipeline} (confidence={conf:.2f}) — {reason}",
            name=preprocessed.source_path.name,
            pipeline=result.pipeline.value,
            conf=result.confidence,
            reason=result.reason,
        )
        return result

    def route_path(
        self,
        path: Path,
        *,
        is_scanned_hint: bool = False,
    ) -> RoutingResult:
        """
        Convenience method that works directly from a file path.

        For non-PDF files: pure extension routing, zero I/O.
        For PDFs: reads basic metadata via fitz (page count, encryption) but
        does NOT run poppler, so is_scanned defaults to False unless the
        caller passes is_scanned_hint=True. Prefer route() when you have a
        PreprocessedPDF for accurate results.
        """
        ext = path.suffix.lower()

        if ext != ".pdf":
            profile = ContentProfile(
                extension=ext,
                page_count=1,
                is_scanned=False,
                encrypted=False,
                page_size="",
                detection_level="none",
            )
            return RoutingRules.route(profile)

        # PDF: read basic metadata via fitz
        encrypted = False
        page_count = 1
        try:
            import fitz  # type: ignore[import-untyped]
            doc = fitz.open(str(path))
            encrypted = doc.is_encrypted
            page_count = doc.page_count
            doc.close()
        except Exception:
            pass

        metadata = PDFMetadata(
            page_count=page_count,
            encrypted=encrypted,
            pdf_version="",
            title="",
            author="",
            page_size="",
            raw={},
        )
        synthetic = PreprocessedPDF(
            source_path=path,
            output_dir=path.parent,
            metadata=metadata,
            is_scanned=is_scanned_hint,
        )
        return self.route(synthetic)
```

## Stage 5: Parsers

All parsers share the `BaseParser` ABC. The only required method is `parse_file(path) → ParsedDocument`. The concrete `parse_folder` method on the base class delegates to `parse_file` for each file it finds matching a glob pattern — so each backend only needs to implement one method.

The **factory function** `create_parser(config)` maps every `PipelineType` to its implementation class via a match statement — one place to change when adding a new backend.

```bash
cat src/corpuscraft/parsers/base.py
```

```output
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from corpuscraft.models import ParsedDocument

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".html", ".htm",
    ".md", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, path: Path) -> ParsedDocument: ...

    def parse_folder(
        self, folder: Path, glob: str = "**/*"
    ) -> list[ParsedDocument]:
        paths = [
            p for p in folder.glob(glob)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return [self.parse_file(p) for p in sorted(paths)]
```

```bash
cat src/corpuscraft/parsers/factory.py
```

```output
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
        case _:
            from corpuscraft.parsers.standard import StandardParser
            return StandardParser(config)
```

All imports inside the factory are deferred (inside the match arms). This means heavy dependencies like Docling, MinerU, and YOLO are only imported when that specific parser is actually requested — not when any other module imports `create_parser`. This keeps startup cost proportional to what you actually use.

### The parsers, from simplest to most complex

**pdfplumber** — Lightest parser. Extracts raw text column by column, detects tables via pdfplumber's built-in table finder, and formats them as plain markdown tables. Good baseline.

```bash
grep -n 'def parse_file\|table\|page\|content' src/corpuscraft/parsers/pdfplumber_parser.py | head -20
```

```output
17:    A parser that uses pdfplumber to extract text and tables from PDFs.
23:    def parse_file(self, path: Path) -> ParsedDocument:
26:        content_parts = []
32:                metadata["page_count"] = len(pdf.pages)
34:                for i, page in enumerate(pdf.pages):
36:                    text = page.extract_text()
38:                        content_parts.append(text)
40:                    # Optionally, you can extract tables. pdfplumber is great at this!
41:                    tables = page.extract_tables()
42:                    for table in tables:
43:                        # Formatting tables as basic markdown
44:                        if not table:
48:                        content_parts.append("\n")
50:                        for row_idx, row in enumerate(table):
54:                            content_parts.append(row_str)
56:                            # Add markdown table separator after header
59:                                content_parts.append(separator)
61:                        content_parts.append("\n")
63:            md_text = "\n".join(content_parts)
66:                content=md_text,
```

**pymupdf** — Next step up. Uses `pymupdf4llm` which calls PyMuPDF's markdown export. Produces high-quality markdown with headers, bold, italics and basic table formatting in a single API call — much faster than any ML-based parser.

```bash
sed -n '1,45p' src/corpuscraft/parsers/pymupdf_parser.py
```

```output
from __future__ import annotations

import logging
from pathlib import Path

import pymupdf4llm
import fitz

from corpuscraft.config import ParserConfig
from corpuscraft.models import ParsedDocument
from corpuscraft.parsers.base import BaseParser

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """
    A pure PyMuPDF parser that leverages pymupdf4llm for markdown conversion.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def parse_file(self, path: Path) -> ParsedDocument:
        logger.info(f"Parsing {path.name} with PyMuPDF")
        
        # We can use pymupdf4llm to easily extract high-quality markdown
        try:
            try:
                md_text = pymupdf4llm.to_markdown(str(path))
            except AttributeError as e:
                # Fallback if RapidOCR integration fails due to dependency conflicts (e.g. text_detector)
                logger.warning(f"PyMuPDF OCR failed, falling back to use_ocr=0: {e}")
                md_text = pymupdf4llm.to_markdown(str(path), use_ocr=0)
            
            # Optional: Extract basic metadata using pure fitz (PyMuPDF)
            metadata = {}
            with fitz.open(str(path)) as doc:
                metadata.update(doc.metadata)
                metadata["page_count"] = doc.page_count
            
            return ParsedDocument(
                content=md_text,
                source_path=path,
                pipeline="pymupdf",
```

**standard (Docling)** — Uses the `DocumentConverter` from Docling, which runs a full document understanding pipeline including layout analysis, reading order detection, and markdown export. Supports PDFs, DOCX, PPTX, HTML, and images. The `gpu` pipeline is the same class with `use_gpu=True`, which enables CUDA acceleration for the underlying vision models.

```bash
grep -n 'def parse_file\|DocumentConverter\|PdfPipeline\|AcceleratorOptions\|num_threads\|format_to_text\|export_to_markdown' src/corpuscraft/parsers/standard.py
```

```output
5:from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
7:from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
8:from docling.document_converter import DocumentConverter, PdfFormatOption
9:from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
20:            pipeline_options = ThreadedPdfPipelineOptions(
21:                accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
23:            self._converter = DocumentConverter(
26:                        pipeline_cls=ThreadedStandardPdfPipeline,
32:            self._converter = DocumentConverter()
34:    def parse_file(self, path: Path) -> ParsedDocument:
37:        content = doc.export_to_markdown()
```

**ocr** — Inherits from `StandardParser` conceptually but uses Docling's OCR pipeline instead. The OCR engine is pluggable: `rapidocr` (default, no system deps), `tesseract`, or `easyocr`. Language codes are normalised — Tesseract uses 3-letter codes (`eng`), EasyOCR uses 2-letter (`en`), and the parser maps between them automatically. Table structure detection is always enabled.

```bash
grep -n 'def parse_file\|ocr_engine\|EasyOcr\|RapidOcr\|Tesseract\|_LANG_MAP\|TableStructure' src/corpuscraft/parsers/ocr.py | head -25
```

```output
14:# EasyOCR uses 2-letter codes; Tesseract uses 3-letter codes.
28:        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
29:        return TesseractCliOcrOptions(force_full_page_ocr=True, lang=languages)
31:        from docling.datamodel.pipeline_options import EasyOcrOptions
32:        return EasyOcrOptions(force_full_page_ocr=True, lang=_to_easyocr_langs(languages))
34:    from docling.datamodel.pipeline_options import RapidOcrOptions
35:    return RapidOcrOptions(force_full_page_ocr=True)
46:            config.ocr_engine, config.ocr_languages
54:    def parse_file(self, path: Path) -> ParsedDocument:
60:            "ocr_engine": self._config.ocr_engine,
```

**vlm** — Vision Language Model parser. Converts every PDF page to an image, then sends each image to an Ollama endpoint (`/v1/chat/completions`) with a prompt asking for structured markdown extraction. The default model is `gabegoodhart/granite-docling:258M`, a compact vision model fine-tuned specifically for document extraction. Requires Ollama running locally.

```bash
grep -n 'def parse_file\|def _check_ollama\|def _pdf_to_images\|def _image_to_markdown\|vlm_host\|base64\|requests.post' src/corpuscraft/parsers/vlm.py | head -20
```

```output
34:        endpoint = f"{config.vlm_host}/v1/chat/completions"
55:    def parse_file(self, path: Path) -> ParsedDocument:
56:        if not check_ollama_connection(self._config.vlm_host):
58:                f"Cannot reach Ollama at {self._config.vlm_host}. "
```

```bash
sed -n '55,100p' src/corpuscraft/parsers/vlm.py
```

```output
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
```

**yolo** — Layout-aware parser. DocLayout-YOLO detects 11 object classes on rendered page images (Title, Section-header, Text, Table, Figure, Caption, Footnote, Formula, List-item, Page-header, Page-footer). For each detected text region, pdfplumber extracts the underlying text at that bounding box. Tables and figures get special formatting; headers and footers are discarded. Pages are rendered at 150 DPI (scale = 150/72 from 72 DPI PDF points). Limitation: scanned PDFs yield no text because pdfplumber needs native text.

```bash
grep -n 'SKIP_LABELS\|_SCALE\|sort.*key\|_extract_text_in_bbox\|def parse_file\|def _process_page' src/corpuscraft/parsers/yolo.py | head -20
```

```output
14:_SCALE = _RENDER_DPI / 72.0  # pixels per PDF point
16:_SKIP_LABELS = {"Page-header", "Page-footer"}
65:    def parse_file(self, path: Path) -> ParsedDocument:
74:                bitmap = pdf_doc[idx].render(scale=_SCALE)
101:                    if label in _SKIP_LABELS:
104:                    bbox = (x1 / _SCALE, y1 / _SCALE, x2 / _SCALE, y2 / _SCALE)
```

```bash
sed -n '65,115p' src/corpuscraft/parsers/yolo.py
```

```output
    def parse_file(self, path: Path) -> ParsedDocument:
        pdf_doc = pdfium.PdfDocument(str(path))
        num_pages = len(pdf_doc)
        page_texts: list[str] = []

        with pdfplumber.open(path) as pdf:
            for idx in range(num_pages):
                plumber_page = pdf.pages[idx]

                bitmap = pdf_doc[idx].render(scale=_SCALE)
                pil_image = bitmap.to_pil()

                results = self._model.predict(
                    pil_image, imgsz=1024, conf=self._conf, device="cpu", verbose=False
                )

                boxes = results[0].boxes if results else None
                if boxes is None or len(boxes) == 0:
                    text = plumber_page.extract_text() or ""
                    if text.strip():
                        page_texts.append(text.strip())
                    else:
                        logger.warning(
                            f"Page {idx + 1} of {path.name}: no detections and no "
                            "extractable text — scanned PDF? Use the ocr pipeline."
                        )
                    continue

                detections = sorted(
                    zip(boxes.cls.tolist(), boxes.xyxy.tolist()),
                    key=lambda d: (d[1][1], d[1][0]),  # y1, x1
                )

                segments: list[str] = []
                for cls_id, (x1, y1, x2, y2) in detections:
                    label = _NAMES.get(int(cls_id), "Text")
                    if label in _SKIP_LABELS:
                        continue

                    bbox = (x1 / _SCALE, y1 / _SCALE, x2 / _SCALE, y2 / _SCALE)
                    text = plumber_page.crop(bbox).extract_text() or ""
                    if not text.strip():
                        continue

                    if label in _HEADING_LABELS:
                        segments.append(f"{_HEADING_LABELS[label]} {text.strip()}")
                    elif label in _BLOCK_LABELS:
                        segments.append(f"{_BLOCK_LABELS[label]}\n{text.strip()}")
                    else:
                        segments.append(text.strip())

```

**mineru** — Calls MinerU's Python API directly (no subprocess). MinerU specialises in academic papers: it handles multi-column layouts, LaTeX formulas, and dense tables. The integration uses a callback mechanism to intercept MinerU's intermediate 'middle JSON' representation before it is converted to markdown — this lets the code capture richer structural data. A `_NullWriter` discards extracted images since they are not needed for text training data.

```bash
grep -n 'class _NullWriter\|def parse_file\|middle_json\|callback\|do_parse\|union_make' src/corpuscraft/parsers/mineru.py | head -20
```

```output
12:class _NullWriter:
23:    The on_doc_ready callback intercepts the intermediate 'middle JSON'
34:    def parse_file(self, path: Path) -> ParsedDocument:
37:            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import make_blocks_to_markdown
50:            middle_json: dict,
53:            for page in middle_json.get("pdf_info", []):
```

```bash
sed -n '34,80p' src/corpuscraft/parsers/mineru.py
```

```output
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
```

**consensus** — The most expensive parser. It runs Docling, YOLO, and MinerU sequentially (not in parallel — PyTorch's import lock makes true parallelism unsafe) and merges their outputs. The merge strategy: use MinerU's text unconditionally, but only include tables and figures if at least `consensus_min_votes` parsers agree the element exists, measured by intersection-over-minimum (IoM) of bounding boxes. This catches structural elements one parser might miss while rejecting hallucinations.

```bash
grep -n 'def _iom\|def _best_iom\|consensus_min_votes\|_run_docling\|_run_yolo\|_run_mineru\|def parse_file' src/corpuscraft/parsers/consensus.py | head -20
```

```output
21:def _iom(a: Bbox, b: Bbox) -> float:
47:def _best_iom(candidate: Bbox, others: list[Bbox]) -> float:
103:def _run_docling(path: Path) -> dict[int, list[Bbox]]:
138:def _run_yolo(path: Path, conf: float, model_id: str) -> dict[int, list[Bbox]]:
169:def _run_mineru(path: Path) -> dict[int, list[dict]]:
212:    - Tables:     MinerU HTML→markdown, only when ≥consensus_min_votes other
223:    def parse_file(self, path: Path) -> ParsedDocument:
237:        min_votes = self.config.consensus_min_votes
244:        docling_boxes: dict[int, list[Bbox]] = _run_docling(path)
245:        yolo_boxes: dict[int, list[Bbox]] = _run_yolo(
248:        mineru_pages: dict[int, list[dict]] = _run_mineru(path)
316:                "consensus_min_votes": min_votes,
```

```bash
sed -n '21,55p' src/corpuscraft/parsers/consensus.py
```

```output
def _iom(a: Bbox, b: Bbox) -> float:
    """Intersection over Minimum — handles size-mismatched boxes from different parsers."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


def _to_topleft_pts(
    bbox: tuple,
    *,
    scale: float = 1.0,
    flip_y: bool = False,
    page_h: float = 0.0,
) -> Bbox:
    x1, y1, x2, y2 = (v / scale for v in bbox)
    if flip_y:
        y1, y2 = page_h - y2, page_h - y1
    return (x1, y1, x2, y2)


def _best_iom(candidate: Bbox, others: list[Bbox]) -> float:
    return max((_iom(candidate, o) for o in others), default=0.0)


# ── HTML table → markdown ────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
```

IoM (intersection over minimum) is used instead of the standard IoU (intersection over union) because bounding boxes from different parsers are often different sizes for the same element — Docling might annotate a tight cell boundary while YOLO draws a box over the entire table. IoM measures overlap relative to the *smaller* box, so a small box fully inside a large one scores 1.0. IoU would penalise that.

## Stage 6: Generation

`QAGenerator` takes a `ParsedDocument` and produces a list of `QAExample` objects. It:

1. Splits `document.content` into overlapping chunks (1500 chars, 150 overlap) using LangChain's `RecursiveCharacterTextSplitter`
2. Distributes the requested example count evenly across chunks (`ceil(n / chunks)`)
3. Calls Ollama with a JSON-requesting system prompt for each chunk
4. Parses the JSON response into `QAExample` objects with question, answer, context, and difficulty

JSON parsing errors are swallowed per-chunk so a bad LLM response on one chunk doesn't abort the entire document.

```bash
cat src/corpuscraft/generators/qa.py
```

```output
from __future__ import annotations

import json
import math

from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpuscraft.config import GeneratorConfig, LLMConfig
from corpuscraft.generators.base import BaseGenerator
from corpuscraft.models import ParsedDocument, QAExample

_SYSTEM_PROMPT = """\
You are an expert at creating question-answer pairs for training AI models.
Given a passage of text, generate diverse question-answer pairs that cover:
- Factual questions (specific facts directly stated in the text)
- Reasoning questions (require inference or synthesis)
- Clarification questions (about terminology or concepts)

Return a JSON array of objects with keys: "question", "answer", "difficulty".
Difficulty must be one of: "easy", "medium", "hard".
Return ONLY the JSON array, no other text."""

_USER_TEMPLATE = """\
Generate {n} question-answer pairs from the following passage:

{passage}"""


class QAGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig, llm: LLMConfig) -> None:
        super().__init__(config, llm)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
        )

    def generate(self, document: ParsedDocument) -> list[QAExample]:
        import ollama

        chunks = self._splitter.split_text(document.content)
        if not chunks:
            return []

        per_chunk = max(1, math.ceil(self.config.num_examples / len(chunks)))
        examples: list[QAExample] = []

        client = ollama.Client(host=self.llm.base_url)

        for chunk in chunks:
            if len(examples) >= self.config.num_examples:
                break
            try:
                response = client.chat(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _USER_TEMPLATE.format(
                                n=per_chunk, passage=chunk
                            ),
                        },
                    ],
                    options={"temperature": self.llm.temperature},
                )
                raw = response.message.content.strip()
                pairs = json.loads(raw)
                for pair in pairs:
                    examples.append(
                        QAExample(
                            question=pair["question"],
                            answer=pair["answer"],
                            context=chunk,
                            source=str(document.source_path),
                            difficulty=pair.get("difficulty", "medium"),
                        )
                    )
            except (json.JSONDecodeError, KeyError, Exception):
                continue

        return examples[: self.config.num_examples]
```

## Stage 7: Export

The exporter shuffles all examples, splits them by the configured ratios (default 80/10/10), and writes one JSONL file per split. Each line is a JSON object representing one `QAExample`. The function returns a dict mapping split name → output path.

```bash
cat src/corpuscraft/exporters/jsonl.py
```

```output
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from corpuscraft.config import ExporterConfig
from corpuscraft.models import QAExample


def export_jsonl(examples: list[QAExample], config: ExporterConfig) -> dict[str, Path]:
    if not examples:
        return {}

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shuffled = list(examples)
    random.shuffle(shuffled)

    train_r, val_r, _ = config.split_ratio
    n = len(shuffled)
    train_end = int(n * train_r)
    val_end = train_end + int(n * val_r)

    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    written: dict[str, Path] = {}
    for name, split_examples in splits.items():
        if not split_examples:
            continue
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
        written[name] = path

    return written
```

## The Full Chain

The `examples/basic/preprocess_then_parse.py` example shows how the five stages compose. It is the canonical entry point for understanding end-to-end usage.

```bash
cat examples/basic/preprocess_then_parse.py
```

```output
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
```

Reading order in the example:

1. `PopplerPreprocessor(clean=True).run(path, out_dir)` → `PreprocessedPDF` (strips annotations, detects if scanned)
2. `PipelineRouter(detection_level='basic').route(preprocessed)` → `RoutingResult` (opens PDF with fitz, scores complexity, picks pipeline)
3. `create_parser(ParserConfig(pipeline=result.pipeline))` → the right `BaseParser` subclass
4. `parser.parse_file(preprocessed.parser_input())` → `ParsedDocument` (uses cleaned PDF path)
5. (not shown) `QAGenerator(...).generate(doc)` → list of `QAExample`
6. (not shown) `export_jsonl(examples, config)` → `{"train": Path, "val": Path, "test": Path}`

The `preprocessed.parser_input()` call in step 4 is the key handshake between preprocessing and parsing: it abstracts over whether cleaning ran or not, so parsers never touch the original file directly.

## Public API surface (`__init__.py`)

Only a small set of symbols is re-exported at the package level — the config types, the two data models, and the parser factory. Everything else (preprocessor, router, generator, exporter) is imported directly from its submodule.

```bash
cat src/corpuscraft/__init__.py
```

```output
from corpuscraft.config import CorpusCraftConfig, load_config, save_default_config
from corpuscraft.models import ParsedDocument, QAExample
from corpuscraft.parsers import create_parser

__all__ = [
    "CorpusCraftConfig",
    "load_config",
    "save_default_config",
    "ParsedDocument",
    "QAExample",
    "create_parser",
]
```

## Summary: data flows and contracts

Each stage consumes one type and produces another. The contracts are narrow by design — stages only depend on their immediate input type, not on how it was produced:

| Stage | Input | Output |
|---|---|---|
| `PopplerPreprocessor.run` | `Path` (PDF) | `PreprocessedPDF` |
| `PipelineRouter.route` | `PreprocessedPDF` | `RoutingResult` |
| `create_parser` + `parse_file` | `ParserConfig` + `Path` | `ParsedDocument` |
| `QAGenerator.generate` | `ParsedDocument` | `list[QAExample]` |
| `export_jsonl` | `list[QAExample]` + `ExporterConfig` | `dict[str, Path]` |

You can skip or replace any stage. Run only the parser without preprocessing. Skip routing and hard-code a pipeline. Use the exporter without a generator by providing your own `QAExample` list. The types are plain dataclasses — nothing locks you into the full chain.
