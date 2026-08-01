from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from corpuscraft.pdf2code.models import BBox, DiffResult, LayoutNode, LayoutTree, NodeDiff

# Standard SSIM constants (k1=0.01, k2=0.03, dynamic range L=255).
_C1 = (0.01 * 255) ** 2
_C2 = (0.03 * 255) ** 2


def diff_page(
    *,
    original_image: Path,
    rendered_image: Path,
    layout_tree: LayoutTree,
    dpi: int,
    out_dir: Path,
    worst_n: int = 10,
) -> DiffResult:
    original = _load_gray(original_image)
    rendered = _load_gray(rendered_image)
    original, rendered = _match_shape(original, rendered)

    global_score = _similarity(original, rendered)
    global_mae = float(np.abs(original - rendered).mean())

    node_diffs: list[NodeDiff] = []
    for node in _leaves(layout_tree.root):
        crop_a = _crop(original, node.bbox, dpi)
        crop_b = _crop(rendered, node.bbox, dpi)
        if crop_a.size == 0 or crop_b.size == 0:
            continue
        crop_a, crop_b = _match_shape(crop_a, crop_b)
        node_diffs.append(
            NodeDiff(
                node_id=node.id,
                bbox=node.bbox,
                score=_similarity(crop_a, crop_b),
                mae=float(np.abs(crop_a - crop_b).mean()),
            )
        )
    node_diffs.sort(key=lambda n: n.score)  # least similar (worst) first

    out_dir.mkdir(parents=True, exist_ok=True)
    viz_path = out_dir / f"diff_page{layout_tree.page_index}.png"
    _render_visualization(original, rendered, node_diffs[:worst_n], dpi, viz_path)

    return DiffResult(
        global_score=global_score,
        global_mae=global_mae,
        node_diffs=node_diffs,
        visualization_path=viz_path,
    )


def _leaves(node: LayoutNode) -> Iterator[LayoutNode]:
    if not node.children:
        yield node
        return
    for child in node.children:
        yield from _leaves(child)


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _match_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    return a[:height, :width], b[:height, :width]


def _crop(arr: np.ndarray, bbox: BBox, dpi: int) -> np.ndarray:
    scale = dpi / 72
    x0, y0, x1, y1 = bbox.scaled(scale).as_tuple()
    height, width = arr.shape
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(width, int(round(x1))), min(height, int(round(y1)))
    if xi1 <= xi0 or yi1 <= yi0:
        return arr[0:0, 0:0]
    return arr[yi0:yi1, xi0:xi1]


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Global (single-window) SSIM: treats the whole array as one region.

    A deliberately simple stand-in for windowed/Gaussian SSIM — no scipy
    dependency, and per-node crops are already small single regions so a
    sliding window buys little there. Revisit if the global page score needs
    to be more sensitive to localized error.
    """
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + _C1) * (2 * cov + _C2)
    denominator = (mu_a**2 + mu_b**2 + _C1) * (var_a + var_b + _C2)
    return float(numerator / denominator)


def _render_visualization(
    original: np.ndarray,
    rendered: np.ndarray,
    worst: list[NodeDiff],
    dpi: int,
    out_path: Path,
) -> None:
    diff = np.abs(original - rendered)
    intensity = np.clip(diff, 0, 255).astype(np.uint8)
    heat = np.zeros((*diff.shape, 3), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = 255 - intensity
    heat[..., 2] = 255 - intensity

    orig_img = Image.fromarray(original.astype(np.uint8)).convert("RGB")
    rend_img = Image.fromarray(rendered.astype(np.uint8)).convert("RGB")
    heat_img = Image.fromarray(heat)

    draw = ImageDraw.Draw(heat_img)
    scale = dpi / 72
    for node in worst:
        x0, y0, x1, y1 = node.bbox.scaled(scale).as_tuple()
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    width = orig_img.width + rend_img.width + heat_img.width
    height = max(orig_img.height, rend_img.height, heat_img.height)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(orig_img, (0, 0))
    canvas.paste(rend_img, (orig_img.width, 0))
    canvas.paste(heat_img, (orig_img.width + rend_img.width, 0))
    canvas.save(out_path)
