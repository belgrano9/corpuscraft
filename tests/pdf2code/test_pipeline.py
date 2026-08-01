from __future__ import annotations

import json
from pathlib import Path

from corpuscraft.pdf2code.pipeline import run_skeleton


def test_run_skeleton_dumps_json_for_every_stage(sample_pdf: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    results = run_skeleton(sample_pdf, out_dir, dpi=100)

    assert len(results) == 1
    assert (out_dir / "extraction.json").exists()
    assert (out_dir / "emitted.html").exists()
    assert (out_dir / "emitted.css").exists()
    assert (out_dir / "render_result.json").exists()
    assert (out_dir / "layout_page0.json").exists()
    assert (out_dir / "diff_page0.json").exists()

    diff_data = json.loads((out_dir / "diff_page0.json").read_text())
    assert "global_score" in diff_data
    assert "node_diffs" in diff_data
