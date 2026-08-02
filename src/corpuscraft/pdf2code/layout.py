from __future__ import annotations

from pathlib import Path

import fitz

from corpuscraft.pdf2code.models import BBox, LayoutNode, LayoutTree, PageExtraction, Primitive


def build_layout_tree(
    page: PageExtraction,
    pdf_path: Path,
    *,
    min_row_gap_pt: float = 4.0,
    min_col_gap_pt: float = 10.0,
    line_overlap_ratio: float = 0.5,
) -> LayoutTree:
    """Recursive XY-cut on whitespace, with a find_tables() pre-pass for grids.

    find_tables() needs a live fitz.Page (it reads the raw content stream,
    not our primitive list), so this re-opens pdf_path rather than taking
    only the PageExtraction produced by extract.py.
    """
    page_id = f"page{page.page_index}"
    consumed: set[int] = set()
    table_nodes = _extract_tables(page, pdf_path, page_id, consumed)

    remaining = [i for i in range(len(page.primitives)) if i not in consumed]
    cutter = _Cutter(page.primitives, min_row_gap_pt, min_col_gap_pt, line_overlap_ratio)
    content_node = cutter.cut(f"{page_id}/content", remaining) if remaining else None

    root = LayoutNode(id=page_id, kind="page", bbox=BBox(0, 0, page.width, page.height))
    root.children.extend(table_nodes)
    if content_node is not None:
        root.children.append(content_node)

    return LayoutTree(page_index=page.page_index, root=root)


def _extract_tables(
    page: PageExtraction, pdf_path: Path, page_id: str, consumed: set[int]
) -> list[LayoutNode]:
    doc = fitz.open(pdf_path)
    try:
        finder = doc[page.page_index].find_tables()
        table_nodes = []
        for table_index, table in enumerate(finder.tables):
            table_id = f"{page_id}/table{table_index}"
            row_nodes = []
            for row_index, row in enumerate(table.rows):
                row_id = f"{table_id}/row{row_index}"
                cell_nodes = []
                for cell_index, cell_bbox in enumerate(row.cells):
                    if cell_bbox is None:
                        continue
                    bbox = BBox(*cell_bbox)
                    refs = [
                        i
                        for i, p in enumerate(page.primitives)
                        if i not in consumed and _center_inside(p.bbox, bbox)
                    ]
                    consumed.update(refs)
                    cell_nodes.append(
                        LayoutNode(id=f"{row_id}/cell{cell_index}", kind="cell", bbox=bbox, primitive_refs=refs)
                    )
                row_nodes.append(
                    LayoutNode(
                        id=row_id,
                        kind="row",
                        bbox=_union_bbox([c.bbox for c in cell_nodes]) if cell_nodes else BBox(0, 0, 0, 0),
                        children=cell_nodes,
                    )
                )
            table_nodes.append(LayoutNode(id=table_id, kind="table", bbox=BBox(*table.bbox), children=row_nodes))
        return table_nodes
    finally:
        doc.close()


def _center_inside(bbox: BBox, container: BBox) -> bool:
    cx, cy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
    return container.x0 <= cx <= container.x1 and container.y0 <= cy <= container.y1


def _union_bbox(bboxes: list[BBox]) -> BBox:
    return BBox(
        min(b.x0 for b in bboxes),
        min(b.y0 for b in bboxes),
        max(b.x1 for b in bboxes),
        max(b.y1 for b in bboxes),
    )


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _largest_gap(intervals: list[tuple[float, float]], min_gap: float) -> tuple[float, float] | None:
    """Largest whitespace gap strictly between two content intervals.

    Only interior gaps count (not leading/trailing margins) — those aren't
    a structural split, there's nothing on one side of them.
    """
    merged = _merge_intervals(intervals)
    best: tuple[float, float] | None = None
    for (_, end), (start, _) in zip(merged, merged[1:]):
        width = start - end
        if width >= min_gap and (best is None or width > best[0]):
            best = (width, (end + start) / 2)
    return best


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    shorter = min(a.height, b.height)
    return overlap / shorter if shorter > 0 else 0.0


class _Cutter:
    def __init__(
        self, primitives: list[Primitive], min_row_gap_pt: float, min_col_gap_pt: float, line_overlap_ratio: float
    ) -> None:
        self._primitives = primitives
        self._min_row_gap = min_row_gap_pt
        self._min_col_gap = min_col_gap_pt
        self._line_overlap_ratio = line_overlap_ratio

    def cut(self, node_id: str, indices: list[int]) -> LayoutNode | None:
        if not indices:
            return None
        if len(indices) == 1:
            return self._leaf(node_id, indices[0])

        y_intervals = [(self._primitives[i].bbox.y0, self._primitives[i].bbox.y1) for i in indices]
        x_intervals = [(self._primitives[i].bbox.x0, self._primitives[i].bbox.x1) for i in indices]
        row_gap = _largest_gap(y_intervals, self._min_row_gap)
        col_gap = _largest_gap(x_intervals, self._min_col_gap)

        if row_gap and (col_gap is None or row_gap[0] >= col_gap[0]):
            node = self._split(node_id, indices, axis="y", coordinate=row_gap[1], kind="block", prefix="r")
            if node is not None:
                return node
        if col_gap:
            node = self._split(node_id, indices, axis="x", coordinate=col_gap[1], kind="column", prefix="c")
            if node is not None:
                return node

        return self._group_lines(node_id, indices)

    def _split(
        self, node_id: str, indices: list[int], *, axis: str, coordinate: float, kind: str, prefix: str
    ) -> LayoutNode | None:
        if axis == "y":
            first = [i for i in indices if self._primitives[i].bbox.y1 <= coordinate]
            second = [i for i in indices if self._primitives[i].bbox.y0 >= coordinate]
        else:
            first = [i for i in indices if self._primitives[i].bbox.x1 <= coordinate]
            second = [i for i in indices if self._primitives[i].bbox.x0 >= coordinate]
        if len(first) + len(second) != len(indices) or not first or not second:
            return None  # something straddles the gap -- fall through to the next strategy

        children = [c for c in (self.cut(f"{node_id}/{prefix}0", first), self.cut(f"{node_id}/{prefix}1", second)) if c]
        bbox = _union_bbox([self._primitives[i].bbox for i in indices])
        return LayoutNode(id=node_id, kind=kind, bbox=bbox, children=children)

    def _leaf(self, node_id: str, index: int) -> LayoutNode:
        primitive = self._primitives[index]
        kind = "line" if primitive.kind == "text" else primitive.kind
        return LayoutNode(id=node_id, kind=kind, bbox=primitive.bbox, primitive_refs=[index])

    def _group_lines(self, node_id: str, indices: list[int]) -> LayoutNode:
        ordered = sorted(indices, key=lambda i: self._primitives[i].bbox.y0)
        groups: list[list[int]] = []
        for i in ordered:
            bbox = self._primitives[i].bbox
            for group in groups:
                group_bbox = _union_bbox([self._primitives[j].bbox for j in group])
                if _y_overlap_ratio(bbox, group_bbox) >= self._line_overlap_ratio:
                    group.append(i)
                    break
            else:
                groups.append([i])

        line_nodes = []
        for line_index, group in enumerate(groups):
            group.sort(key=lambda i: self._primitives[i].bbox.x0)
            line_id = f"{node_id}/line{line_index}"
            if len(group) == 1:
                line_nodes.append(self._leaf(line_id, group[0]))
            else:
                bbox = _union_bbox([self._primitives[j].bbox for j in group])
                line_nodes.append(LayoutNode(id=line_id, kind="line", bbox=bbox, primitive_refs=group))

        if len(line_nodes) == 1:
            return line_nodes[0]
        bbox = _union_bbox([self._primitives[i].bbox for i in indices])
        return LayoutNode(id=node_id, kind="block", bbox=bbox, children=line_nodes)
