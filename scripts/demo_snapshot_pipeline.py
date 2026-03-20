#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    py = sys.executable
    out_dir = PROJECT_ROOT / "output" / "demo_snapshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "run_snapshot_calibration.py"),
            "--ticker",
            "SPY",
            "--date",
            "2026-01-04",
            "--out",
            str(out_dir),
            "--input_file",
            str(PROJECT_ROOT / "data" / "raw" / "spy_options_snapshot.parquet"),
            "--option_type",
            "call",
            "--risk_free_rate",
            "0.04",
            "--dividend_yield",
            "0.0",
        ]
    )

    run_cmd(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "run_pricing_validation.py"),
            "--calib_dir",
            str(out_dir),
        ]
    )

    run_cmd(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "generate_report.py"),
            "--output-dir",
            str(out_dir),
        ]
    )

    print(f"Demo artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
