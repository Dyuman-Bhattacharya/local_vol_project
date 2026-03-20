from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_option_panel_cboe_smoke(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "Quote Date": "2026-03-16",
                "Underlying Symbol": "SPY",
                "Root": "SPY",
                "Expiration": "2026-03-20",
                "Strike": 560.0,
                "Option Type": "C",
                "Bid 1545": 5.10,
                "Ask 1545": 5.30,
                "Underlying Bid 1545": 558.9,
                "Underlying Ask 1545": 559.1,
                "Active Underlying Price 1545": 559.0,
                "Trade Volume": 1000,
                "Open Interest": 5000,
            },
            {
                "Quote Date": "2026-03-17",
                "Underlying Symbol": "SPY",
                "Root": "SPY",
                "Expiration": "2026-03-21",
                "Strike": 561.0,
                "Option Type": "P",
                "Bid 1545": 4.10,
                "Ask 1545": 4.30,
                "Underlying Bid 1545": 560.9,
                "Underlying Ask 1545": 561.1,
                "Active Underlying Price 1545": 561.0,
                "Trade Volume": 900,
                "Open Interest": 4500,
            },
        ]
    )
    input_csv = tmp_path / "cboe_sample.csv"
    out_csv = tmp_path / "spy_panel.csv"
    raw.to_csv(input_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/build_option_panel_cboe.py",
            "--input",
            str(input_csv),
            "--out",
            str(out_csv),
            "--underlying",
            "SPY",
            "--snapshot",
            "1545",
            "--risk_free_rate",
            "0.04",
            "--dividend_yield",
            "0.0",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    assert out_csv.exists()
    panel = pd.read_csv(out_csv)
    assert len(panel) == 2
    summary = json.loads((tmp_path / "spy_panel_summary.json").read_text())
    assert summary["dates"] == 2
    assert summary["underlyings"] == ["SPY"]
