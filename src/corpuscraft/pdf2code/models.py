from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def intersect(self, other: BBox) -> BBox | None:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(x0, y0, x1, y1)

    def iou(self, other: BBox) -> float:
        inter = self.intersect(other)
        if inter is None:
            return 0.0
        union = self.area + other.area - inter.area
        return inter.area / union if union > 0 else 0.0

    def scaled(self, factor: float) -> BBox:
        return BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(kw_only=True)
class TextSpan:
    kind: Literal["text"] = "text"
    text: str
    bbox: BBox
    font_family: str
    font_size: float
    font_weight: int
    italic: bool
    color: str
    origin: tuple[float, float]
    page_index: int


@dataclass(kw_only=True)
class PathSegment:
    op: Literal["m", "l", "c", "re", "qu"]
    points: list[tuple[float, float]]


@dataclass(kw_only=True)
class VectorDrawing:
    kind: Literal["drawing"] = "drawing"
    bbox: BBox
    segments: list[PathSegment]
    stroke_color: str | None
    fill_color: str | None
    line_width: float
    page_index: int


@dataclass(kw_only=True)
class ImagePrimitive:
    kind: Literal["image"] = "image"
    bbox: BBox
    image_path: Path
    xref: int
    width_px: int
    height_px: int
    rotation: int
    page_index: int


Primitive = Union[TextSpan, VectorDrawing, ImagePrimitive]


@dataclass(kw_only=True)
class PageExtraction:
    page_index: int
    width: float
    height: float
    rotation: int
    primitives: list[Primitive] = field(default_factory=list)


@dataclass(kw_only=True)
class DocumentExtraction:
    source_path: Path
    pages: list[PageExtraction] = field(default_factory=list)


@dataclass(kw_only=True)
class LayoutNode:
    id: str
    kind: Literal["page", "column", "block", "line", "image", "drawing"]
    bbox: BBox
    children: list[LayoutNode] = field(default_factory=list)
    primitive_refs: list[int] = field(default_factory=list)
    style_class: str | None = None


@dataclass(kw_only=True)
class LayoutTree:
    page_index: int
    root: LayoutNode


@dataclass(kw_only=True)
class NodeDiff:
    node_id: str
    bbox: BBox
    score: float
    mae: float


@dataclass(kw_only=True)
class DiffResult:
    global_score: float
    global_mae: float
    node_diffs: list[NodeDiff] = field(default_factory=list)
    visualization_path: Path | None = None


@dataclass(kw_only=True)
class RenderResult:
    pdf_path: Path
    page_images: list[Path]
    dpi: int
