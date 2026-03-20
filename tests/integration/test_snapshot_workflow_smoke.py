from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_snapshot_workflow_writes_arbitrage_free_iv_surface(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    date = "2026-01-04"
    output_root = tmp_path / "daily_output"

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_snapshot_workflow.py"),
        "--ticker",
        "SPY",
        "--date",
        date,
        "--output_root",
        str(output_root),
        "--input_file",
        str(root / "data" / "raw" / "spy_options_snapshot.parquet"),
        "--option_type",
        "call",
        "--risk_free_rate",
        "0.04",
        "--dividend_yield",
        "0.0",
    ]
    subprocess.run(cmd, cwd=str(root), check=True)

    out_dir = output_root / date
    assert (out_dir / "calibration_summary.json").exists()
    assert (out_dir / "arbitrage_free_iv_surface.pkl").exists()
    assert (out_dir / "arbitrage_free_iv_surface.npz").exists()

    summary = json.loads((out_dir / "calibration_summary.json").read_text())
    assert summary["arbitrage_counts"]["n_fail"] == 0
    assert summary["arbitrage_free_iv_surface_artifact"] is True
