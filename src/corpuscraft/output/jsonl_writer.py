"""JSONL output writer with train/val/test splits."""

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Write dataset to JSONL format with splits."""

    def __init__(
        self,
        output_dir: Path,
        split_ratio: list[float] | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        """Initialize JSONL writer.

        Args:
            output_dir: Output directory
            split_ratio: Train/val/test split ratios (default: [0.8, 0.1, 0.1])
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.split_ratio = split_ratio or [0.8, 0.1, 0.1]
        self.shuffle = shuffle
        self.seed = seed

        # Validate split ratios
        if not abs(sum(self.split_ratio) - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.split_ratio)}")

        if len(self.split_ratio) != 3:
            raise ValueError(f"Must provide 3 split ratios, got {len(self.split_ratio)}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized JSONLWriter: output_dir={output_dir}, split={self.split_ratio}")

    def write(
        self,
        data: list[dict[str, Any]],
        dataset_name: str = "dataset",
    ) -> dict[str, Path]:
        """Write data to JSONL files with train/val/test splits.

        Args:
            data: List of data examples
            dataset_name: Base name for output files

        Returns:
            Dictionary mapping split names to file paths
        """
        if not data:
            logger.warning("No data to write")
            return {}

        logger.info(f"Writing {len(data)} examples to {self.output_dir}")

        # Shuffle if requested
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(data)

        # Calculate split sizes
        n = len(data)
        train_size = int(n * self.split_ratio[0])
        val_size = int(n * self.split_ratio[1])

        # Split data
        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]

        # Write splits
        output_files: dict[str, Path] = {}

        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
        }

        for split_name, split_data in splits.items():
            if not split_data:
                logger.warning(f"No data for {split_name} split")
                continue

            output_path = self.output_dir / f"{dataset_name}_{split_name}.jsonl"
            self._write_jsonl(split_data, output_path)
            output_files[split_name] = output_path

            logger.info(f"Wrote {len(split_data)} examples to {output_path}")

        # Write metadata
        metadata = {
            "total_examples": len(data),
            "splits": {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
            },
            "split_ratio": self.split_ratio,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }

        metadata_path = self.output_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote metadata to {metadata_path}")

        return output_files

    def _write_jsonl(self, data: list[dict[str, Any]], output_path: Path) -> None:
        """Write data to a JSONL file.

        Args:
            data: List of data examples
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def write_single_file(
        self,
        data: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Write data to a single JSONL file without splitting.

        Args:
            data: List of data examples
            output_path: Output file path
        """
        logger.info(f"Writing {len(data)} examples to {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(data, output_path)

        logger.info(f"Successfully wrote {output_path}")

    @staticmethod
    def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
        """Read data from a JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of data examples
        """
        data: list[dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_statistics(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about the dataset.

        Args:
            data: List of data examples

        Returns:
            Dictionary of statistics
        """
        if not data:
            return {}

        stats: dict[str, Any] = {
            "total_examples": len(data),
            "fields": list(data[0].keys()) if data else [],
        }

        # Count field occurrences
        field_counts: dict[str, int] = {}
        for example in data:
            for field in example:
                field_counts[field] = field_counts.get(field, 0) + 1

        stats["field_coverage"] = field_counts

        return stats
