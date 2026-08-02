import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "Usage: uv run examples/pdf2code/inspect_primitives.py "
            "<extraction.json> <page_index> <primitive_index> [primitive_index...]"
        )
        raise SystemExit(1)

    extraction_path = Path(sys.argv[1])
    page_index = int(sys.argv[2])
    indices = [int(i) for i in sys.argv[3:]]

    data = json.loads(extraction_path.read_text())
    page = next(p for p in data["pages"] if p["page_index"] == page_index)

    for index in indices:
        primitive = page["primitives"][index]
        print(f"--- page{page_index}/prim{index} ---")
        print(json.dumps(primitive, indent=2))


if __name__ == "__main__":
    main()
