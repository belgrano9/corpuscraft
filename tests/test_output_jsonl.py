"""Tests for JSONL output writer."""

import json
from pathlib import Path
from typing import Any

import pytest

from corpuscraft.output.jsonl_writer import JSONLWriter


class TestJSONLWriter:
    """Tests for JSONLWriter class."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test JSONLWriter initialization."""
        writer = JSONLWriter(
            output_dir=tmp_path,
            split_ratio=[0.7, 0.2, 0.1],
            shuffle=True,
            seed=42,
        )

        assert writer.output_dir == tmp_path
        assert writer.split_ratio == [0.7, 0.2, 0.1]
        assert writer.shuffle is True
        assert writer.seed == 42
        assert tmp_path.exists()

    def test_initialization_with_defaults(self, tmp_path: Path) -> None:
        """Test initialization with default values."""
        writer = JSONLWriter(output_dir=tmp_path)

        assert writer.split_ratio == [0.8, 0.1, 0.1]
        assert writer.shuffle is True
        assert writer.seed == 42

    def test_invalid_split_ratio_sum(self, tmp_path: Path) -> None:
        """Test that invalid split ratios raise an error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            JSONLWriter(
                output_dir=tmp_path,
                split_ratio=[0.5, 0.3, 0.1],  # Sums to 0.9
            )

    def test_invalid_split_ratio_length(self, tmp_path: Path) -> None:
        """Test that wrong number of split ratios raises an error."""
        with pytest.raises(ValueError, match="Must provide 3 split ratios"):
            JSONLWriter(
                output_dir=tmp_path,
                split_ratio=[0.8, 0.2],  # Only 2 values
            )

    def test_write_empty_data(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test writing empty data."""
        writer = JSONLWriter(output_dir=tmp_path)

        with caplog.at_level("WARNING"):
            result = writer.write([])

        assert result == {}
        assert "No data to write" in caplog.text

    def test_write_splits_data_correctly(self, tmp_path: Path) -> None:
        """Test that data is split correctly."""
        writer = JSONLWriter(
            output_dir=tmp_path,
            split_ratio=[0.6, 0.2, 0.2],
            shuffle=False,  # Don't shuffle for predictable results
        )

        data = [{"id": i, "value": f"item_{i}"} for i in range(10)]

        output_files = writer.write(data, dataset_name="test_dataset")

        # Check that files were created
        assert "train" in output_files
        assert "val" in output_files
        assert "test" in output_files

        # Read and verify split sizes
        train_data = JSONLWriter.read_jsonl(output_files["train"])
        val_data = JSONLWriter.read_jsonl(output_files["val"])
        test_data = JSONLWriter.read_jsonl(output_files["test"])

        assert len(train_data) == 6  # 60% of 10
        assert len(val_data) == 2  # 20% of 10
        assert len(test_data) == 2  # 20% of 10

        # Verify data integrity (no shuffle)
        assert train_data[0]["id"] == 0
        assert val_data[0]["id"] == 6
        assert test_data[0]["id"] == 8

    def test_write_with_shuffle(self, tmp_path: Path) -> None:
        """Test that shuffle works correctly."""
        data = [{"id": i, "value": f"item_{i}"} for i in range(100)]

        # Write with seed
        writer1 = JSONLWriter(output_dir=tmp_path / "run1", shuffle=True, seed=42)
        files1 = writer1.write(data, dataset_name="dataset")
        train_data1 = JSONLWriter.read_jsonl(files1["train"])

        # Write with same seed
        writer2 = JSONLWriter(output_dir=tmp_path / "run2", shuffle=True, seed=42)
        files2 = writer2.write(data, dataset_name="dataset")
        train_data2 = JSONLWriter.read_jsonl(files2["train"])

        # Should be identical
        assert train_data1 == train_data2

        # Write with different seed
        writer3 = JSONLWriter(output_dir=tmp_path / "run3", shuffle=True, seed=123)
        files3 = writer3.write(data, dataset_name="dataset")
        train_data3 = JSONLWriter.read_jsonl(files3["train"])

        # Should be different
        assert train_data1 != train_data3

    def test_write_creates_metadata(self, tmp_path: Path) -> None:
        """Test that metadata file is created."""
        writer = JSONLWriter(
            output_dir=tmp_path,
            split_ratio=[0.7, 0.2, 0.1],
            shuffle=True,
            seed=99,
        )

        data = [{"id": i} for i in range(50)]
        writer.write(data, dataset_name="test_data")

        # Check metadata file
        metadata_path = tmp_path / "test_data_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["total_examples"] == 50
        assert metadata["splits"]["train"] == 35  # 70% of 50
        assert metadata["splits"]["val"] == 10  # 20% of 50
        assert metadata["splits"]["test"] == 5  # 10% of 50
        assert metadata["split_ratio"] == [0.7, 0.2, 0.1]
        assert metadata["shuffle"] is True
        assert metadata["seed"] == 99

    def test_write_single_file(self, tmp_path: Path) -> None:
        """Test writing to a single file without splits."""
        writer = JSONLWriter(output_dir=tmp_path)
        data = [{"id": i, "text": f"sample_{i}"} for i in range(10)]

        output_path = tmp_path / "single_output.jsonl"
        writer.write_single_file(data, output_path)

        assert output_path.exists()

        # Read and verify
        loaded_data = JSONLWriter.read_jsonl(output_path)
        assert len(loaded_data) == 10
        assert loaded_data[0]["id"] == 0
        assert loaded_data[9]["id"] == 9

    def test_read_jsonl(self, tmp_path: Path) -> None:
        """Test reading JSONL files."""
        # Create a test JSONL file
        test_data = [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92},
        ]

        test_file = tmp_path / "test.jsonl"
        with open(test_file, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # Read using the static method
        loaded_data = JSONLWriter.read_jsonl(test_file)

        assert len(loaded_data) == 3
        assert loaded_data[0]["name"] == "Alice"
        assert loaded_data[1]["score"] == 87
        assert loaded_data[2]["id"] == 3

    def test_get_statistics(self, tmp_path: Path) -> None:
        """Test statistics generation."""
        writer = JSONLWriter(output_dir=tmp_path)

        data = [
            {"id": 1, "text": "sample 1", "label": "A"},
            {"id": 2, "text": "sample 2", "label": "B"},
            {"id": 3, "text": "sample 3", "label": "A"},
        ]

        stats = writer.get_statistics(data)

        assert stats["total_examples"] == 3
        assert stats["fields"] == ["id", "text", "label"]
        assert stats["field_coverage"]["id"] == 3
        assert stats["field_coverage"]["text"] == 3
        assert stats["field_coverage"]["label"] == 3

    def test_get_statistics_empty(self, tmp_path: Path) -> None:
        """Test statistics with empty data."""
        writer = JSONLWriter(output_dir=tmp_path)
        stats = writer.get_statistics([])

        assert stats == {}

    def test_get_statistics_with_missing_fields(self, tmp_path: Path) -> None:
        """Test statistics with inconsistent fields."""
        writer = JSONLWriter(output_dir=tmp_path)

        data = [
            {"id": 1, "text": "sample 1"},
            {"id": 2, "text": "sample 2", "extra": "data"},
            {"id": 3},
        ]

        stats = writer.get_statistics(data)

        assert stats["total_examples"] == 3
        assert stats["field_coverage"]["id"] == 3
        assert stats["field_coverage"]["text"] == 2
        assert stats["field_coverage"]["extra"] == 1

    def test_write_with_unicode(self, tmp_path: Path) -> None:
        """Test writing data with Unicode characters."""
        writer = JSONLWriter(output_dir=tmp_path)

        data = [
            {"text": "Hello 世界", "emoji": "🚀"},
            {"text": "Привет мир", "emoji": "🌍"},
            {"text": "مرحبا بالعالم", "emoji": "✨"},
        ]

        output_files = writer.write(data, dataset_name="unicode_test")

        # Read all data back
        all_data = []
        for split_file in output_files.values():
            all_data.extend(JSONLWriter.read_jsonl(split_file))

        # Verify Unicode is preserved
        assert len(all_data) == 3
        texts = [item["text"] for item in all_data]
        assert "Hello 世界" in texts
        assert "Привет мир" in texts
        assert "مرحبا بالعالم" in texts

    def test_write_handles_edge_cases(self, tmp_path: Path) -> None:
        """Test edge cases in data splitting."""
        writer = JSONLWriter(
            output_dir=tmp_path,
            split_ratio=[0.8, 0.1, 0.1],
            shuffle=False,
        )

        # Very small dataset
        small_data = [{"id": 1}, {"id": 2}]
        files = writer.write(small_data, dataset_name="small")

        # Should still create all splits, even if some are empty
        train_data = JSONLWriter.read_jsonl(files["train"])
        val_data = JSONLWriter.read_jsonl(files["val"])
        test_data = JSONLWriter.read_jsonl(files["test"])

        # Check total is preserved
        total = len(train_data) + len(val_data) + len(test_data)
        assert total == 2

    def test_write_file_naming(self, tmp_path: Path) -> None:
        """Test that output files are named correctly."""
        writer = JSONLWriter(output_dir=tmp_path)
        data = [{"id": i} for i in range(20)]

        output_files = writer.write(data, dataset_name="my_dataset")

        assert output_files["train"].name == "my_dataset_train.jsonl"
        assert output_files["val"].name == "my_dataset_val.jsonl"
        assert output_files["test"].name == "my_dataset_test.jsonl"

    def test_output_directory_creation(self, tmp_path: Path) -> None:
        """Test that output directory is created if it doesn't exist."""
        nested_path = tmp_path / "level1" / "level2" / "level3"
        assert not nested_path.exists()

        writer = JSONLWriter(output_dir=nested_path)

        assert nested_path.exists()
        assert nested_path.is_dir()

    def test_jsonl_format_validity(self, tmp_path: Path) -> None:
        """Test that output is valid JSONL format."""
        writer = JSONLWriter(output_dir=tmp_path, shuffle=False)
        data = [
            {"id": 1, "value": "first"},
            {"id": 2, "value": "second"},
            {"id": 3, "value": "third"},
        ]

        output_files = writer.write(data, dataset_name="format_test")

        # Verify each line is valid JSON
        for file_path in output_files.values():
            with open(file_path) as f:
                lines = f.readlines()
                for line in lines:
                    # Each line should be valid JSON
                    parsed = json.loads(line.strip())
                    assert isinstance(parsed, dict)
                    assert "id" in parsed
                    assert "value" in parsed
