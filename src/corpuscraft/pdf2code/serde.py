from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from corpuscraft.pdf2code.models import (
    BBox,
    DiffResult,
    DocumentExtraction,
    ImagePrimitive,
    LayoutNode,
    LayoutTree,
    NodeDiff,
    PageExtraction,
    PathSegment,
    Primitive,
    TextSpan,
    VectorDrawing,
)


class _PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def dump_json(obj: Any, path: Path) -> None:
    payload = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=_PathEncoder))


def _bbox_from_dict(d: dict) -> BBox:
    return BBox(x0=d["x0"], y0=d["y0"], x1=d["x1"], y1=d["y1"])


def _primitive_from_dict(d: dict) -> Primitive:
    kind = d["kind"]
    if kind == "text":
        return TextSpan(
            text=d["text"],
            bbox=_bbox_from_dict(d["bbox"]),
            font_family=d["font_family"],
            font_size=d["font_size"],
            font_weight=d["font_weight"],
            italic=d["italic"],
            color=d["color"],
            origin=tuple(d["origin"]),
            page_index=d["page_index"],
        )
    if kind == "drawing":
        return VectorDrawing(
            bbox=_bbox_from_dict(d["bbox"]),
            segments=[
                PathSegment(op=s["op"], points=[tuple(p) for p in s["points"]])
                for s in d["segments"]
            ],
            stroke_color=d["stroke_color"],
            fill_color=d["fill_color"],
            line_width=d["line_width"],
            page_index=d["page_index"],
        )
    if kind == "image":
        return ImagePrimitive(
            bbox=_bbox_from_dict(d["bbox"]),
            image_path=Path(d["image_path"]),
            xref=d["xref"],
            width_px=d["width_px"],
            height_px=d["height_px"],
            rotation=d["rotation"],
            page_index=d["page_index"],
        )
    raise ValueError(f"unknown primitive kind: {kind!r}")


def _page_from_dict(d: dict) -> PageExtraction:
    return PageExtraction(
        page_index=d["page_index"],
        width=d["width"],
        height=d["height"],
        rotation=d["rotation"],
        primitives=[_primitive_from_dict(p) for p in d["primitives"]],
    )


def dump_document_extraction(doc: DocumentExtraction, path: Path) -> None:
    dump_json(doc, path)


def load_document_extraction(path: Path) -> DocumentExtraction:
    data = json.loads(path.read_text())
    return DocumentExtraction(
        source_path=Path(data["source_path"]),
        pages=[_page_from_dict(p) for p in data["pages"]],
    )


def _node_from_dict(d: dict) -> LayoutNode:
    return LayoutNode(
        id=d["id"],
        kind=d["kind"],
        bbox=_bbox_from_dict(d["bbox"]),
        children=[_node_from_dict(c) for c in d["children"]],
        primitive_refs=list(d["primitive_refs"]),
        style_class=d.get("style_class"),
    )


def dump_layout_tree(tree: LayoutTree, path: Path) -> None:
    dump_json(tree, path)


def load_layout_tree(path: Path) -> LayoutTree:
    data = json.loads(path.read_text())
    return LayoutTree(page_index=data["page_index"], root=_node_from_dict(data["root"]))


def dump_diff_result(result: DiffResult, path: Path) -> None:
    dump_json(result, path)


def load_diff_result(path: Path) -> DiffResult:
    data = json.loads(path.read_text())
    return DiffResult(
        global_score=data["global_score"],
        global_mae=data["global_mae"],
        node_diffs=[
            NodeDiff(node_id=n["node_id"], bbox=_bbox_from_dict(n["bbox"]), score=n["score"], mae=n["mae"])
            for n in data["node_diffs"]
        ],
        visualization_path=Path(data["visualization_path"]) if data.get("visualization_path") else None,
    )
