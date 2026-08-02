import json
import sys
from pathlib import Path

from PIL import Image


def _find_node(node: dict, node_id: str) -> dict | None:
    if node["id"] == node_id:
        return node
    for child in node["children"]:
        found = _find_node(child, node_id)
        if found is not None:
            return found
    return None


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: uv run examples/pdf2code/inspect_node.py "
            "<run_dir> <page_index> <node_id> [zoom] [margin_px]"
        )
        raise SystemExit(1)

    run_dir = Path(sys.argv[1])
    page_index = int(sys.argv[2])
    node_id = sys.argv[3]
    zoom = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    margin = int(sys.argv[5]) if len(sys.argv) > 5 else 15

    render_result = json.loads((run_dir / "render_result.json").read_text())
    scale = render_result["dpi"] / 72

    layout_data = json.loads((run_dir / f"layout_page{page_index}.json").read_text())
    node = _find_node(layout_data["root"], node_id)
    if node is None:
        print(f"node {node_id!r} not found in layout_page{page_index}.json")
        raise SystemExit(1)

    bbox = node["bbox"]
    x0 = int(bbox["x0"] * scale) - margin
    y0 = int(bbox["y0"] * scale) - margin
    x1 = int(bbox["x1"] * scale) + margin
    y1 = int(bbox["y1"] * scale) + margin

    original = Image.open(run_dir / "original_raster" / f"page{page_index}.png").convert("RGB")
    rendered = Image.open(run_dir / "render" / f"page{page_index}.png").convert("RGB")

    def crop(img: Image.Image) -> Image.Image:
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(img.width, x1), min(img.height, y1)
        cropped = img.crop((cx0, cy0, cx1, cy1))
        return cropped.resize((cropped.width * zoom, cropped.height * zoom), Image.NEAREST)

    orig_crop, rend_crop = crop(original), crop(rendered)
    width = orig_crop.width + rend_crop.width + 10
    height = max(orig_crop.height, rend_crop.height)
    canvas = Image.new("RGB", (width, height), (255, 0, 255))
    canvas.paste(orig_crop, (0, 0))
    canvas.paste(rend_crop, (orig_crop.width + 10, 0))

    out_path = run_dir / f"inspect_{node_id.replace('/', '_')}.png"
    canvas.save(out_path)
    print(f"node {node_id}: bbox={bbox}")
    print(f"left=original right=rendered, zoom={zoom}x")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
