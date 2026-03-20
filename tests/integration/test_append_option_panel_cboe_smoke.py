from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_append_option_panel_cboe_smoke(tmp_path: Path) -> None:
    raw1 = pd.DataFrame(
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
            }
        ]
    )
    raw2 = pd.DataFrame(
        [
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
            }
        ]
    )
    day1 = tmp_path / "day1.csv"
    day2 = tmp_path / "day2.csv"
    panel = tmp_path / "panel.parquet"
    raw1.to_csv(day1, index=False)
    raw2.to_csv(day2, index=False)

    root = Path(__file__).resolve().parents[2]

    for src in [day1, day2]:
        subprocess.run(
            [
                sys.executable,
                "scripts/append_option_panel_cboe.py",
                "--input",
                str(src),
                "--panel",
                str(panel),
                "--underlying",
                "SPY",
                "--snapshot",
                "1545",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    df = pd.read_parquet(panel)
    assert len(df) == 2
    assert df["date"].nunique() == 2
