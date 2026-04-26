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
