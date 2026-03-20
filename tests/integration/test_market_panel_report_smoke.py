from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.fixtures.synthetic_data import BSDailyPanelConfig, generate_bs_daily_panel


def test_market_panel_report_smoke(tmp_path: Path) -> None:
    panel, history = generate_bs_daily_panel(BSDailyPanelConfig())
    panel_path = tmp_path / "panel.csv"
    history_path = tmp_path / "history.csv"
    out_dir = tmp_path / "out"
    panel.to_csv(panel_path, index=False)
    history.to_csv(history_path, index=False)

    root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "scripts/run_backtest_market_panel.py",
            "--ticker",
            "SYNTH",
            "--panel_file",
            str(panel_path),
            "--history_csv",
            str(history_path),
            "--out",
            str(out_dir),
            "--option_type",
            "call",
            "--strike_mode",
            "atm",
            "--T",
            str(8.0 / 365.0),
            "--entry_step_days",
            "20",
            "--max_contracts",
            "1",
            "--tx_bps",
            "0.0",
            "--pde_space",
            "80",
            "--pde_time",
            "60",
            "--min_volume",
            "0",
            "--min_open_interest",
            "0",
            "--min_points_per_slice",
            "5",
            "--n_strikes",
            "35",
        ],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_report.py",
            "--output-dir",
            str(out_dir),
        ],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "daily-recalibrated market-panel hedge study" in report
    assert "common market entry premiums" in report
