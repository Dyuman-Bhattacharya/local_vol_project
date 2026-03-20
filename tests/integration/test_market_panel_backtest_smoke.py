from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tests.fixtures.synthetic_data import BSDailyPanelConfig, generate_bs_daily_panel


def test_market_panel_backtest_script_smoke(tmp_path: Path) -> None:
    panel, history = generate_bs_daily_panel(BSDailyPanelConfig())
    panel_path = tmp_path / "panel.csv"
    history_path = tmp_path / "history.csv"
    out_dir = tmp_path / "out"
    panel.to_csv(panel_path, index=False)
    history.to_csv(history_path, index=False)

    proc = subprocess.run(
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
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    assert proc.returncode == 0
    assert (out_dir / "selected_contracts.csv").exists()
    assert (out_dir / "daily_market_backtest_BS.csv").exists()
    assert (out_dir / "daily_market_backtest_LocalVol.csv").exists()
    assert (out_dir / "daily_market_backtest_summary.json").exists()

    summary = json.loads((out_dir / "daily_market_backtest_summary.json").read_text())
    assert summary["n_contracts"] == 1

    bs = pd.read_csv(out_dir / "daily_market_backtest_BS.csv")
    lv = pd.read_csv(out_dir / "daily_market_backtest_LocalVol.csv")
    assert len(bs) == 1
    assert len(lv) == 1
