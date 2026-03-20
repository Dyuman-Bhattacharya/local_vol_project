from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from tests.fixtures.synthetic_data import BSDailyPanelConfig, generate_bs_daily_panel


def test_canonical_daily_update_smoke(tmp_path: Path) -> None:
    panel, _ = generate_bs_daily_panel(BSDailyPanelConfig())
    panel_path = tmp_path / "panel.csv"
    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "panel_manifest.json"
    panel.to_csv(panel_path, index=False)

    cfg = {
        "collection": {
            "ticker": "SYNTH",
            "timezone": "America/New_York",
            "snapshot_time": "15:45",
            "archive_root": str(tmp_path / "archive"),
            "panel_out": str(panel_path),
            "panel_manifest": str(manifest_path),
            "min_days": 1,
            "max_days": 365,
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
        },
        "backtest": {
            "enabled": True,
            "output_dir": str(out_dir),
            "option_type": "call",
            "strike_mode": "atm",
            "T_years": 8.0 / 365.0,
            "entry_step_days": 20,
            "max_contracts": 1,
            "min_panel_dates": 2,
            "tx_bps": 0.0,
            "pde_space": 80,
            "pde_time": 60,
            "min_volume": 0.0,
            "min_open_interest": 0.0,
            "moneyness_min": 0.80,
            "moneyness_max": 1.20,
            "max_relative_spread": 0.20,
            "iv_min": 0.01,
            "iv_max": 1.0,
            "min_points_per_slice": 5,
            "n_strikes": 35,
        },
        "report": {"generate": True},
    }
    cfg_path = tmp_path / "daily_collection.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_canonical_daily_update.py",
            "--config",
            str(cfg_path),
            "--skip_collection",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    assert (out_dir / "daily_market_backtest_summary.json").exists()
    assert (out_dir / "daily_market_backtest_BS.csv").exists()
    assert (out_dir / "daily_market_backtest_LocalVol.csv").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "archive_status.json").exists()
    assert (out_dir / "archive" / "archive_index.json").exists()
    status = json.loads((out_dir / "daily_update_status.json").read_text())
    assert status["canonical_snapshot_time_local"] == "15:45"
    archive_status = json.loads((out_dir / "archive_status.json").read_text())
    assert archive_status["n_panel_dates"] >= 2
    archived_report = Path(archive_status["archive_dir"]) / "report.md"
    assert archived_report.exists()


def test_canonical_daily_update_does_not_fallback_to_sample_snapshot(tmp_path: Path) -> None:
    panel, _ = generate_bs_daily_panel(BSDailyPanelConfig())
    panel_path = tmp_path / "panel.csv"
    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "panel_manifest.json"
    panel.to_csv(panel_path, index=False)

    # Pre-populate stale calibration artifacts that should be cleared.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_summary.json").write_text('{"date": "stale"}', encoding="utf-8")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "01_data_distribution.png").write_bytes(b"stale")

    cfg = {
        "collection": {
            "ticker": "SYNTH",
            "timezone": "America/New_York",
            "snapshot_time": "15:45",
            "archive_root": str(tmp_path / "archive"),
            "panel_out": str(panel_path),
            "panel_manifest": str(manifest_path),
            "min_days": 1,
            "max_days": 365,
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
        },
        "backtest": {
            "enabled": False,
            "output_dir": str(out_dir),
            "option_type": "call",
            "min_volume": 0.0,
            "min_open_interest": 0.0,
            "moneyness_min": 0.80,
            "moneyness_max": 1.20,
            "max_relative_spread": 0.20,
            "iv_min": 0.01,
            "iv_max": 1.0,
            "min_points_per_slice": 50,
            "n_strikes": 35,
        },
        "report": {"generate": False},
    }
    cfg_path = tmp_path / "daily_collection.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_canonical_daily_update.py",
            "--config",
            str(cfg_path),
            "--skip_collection",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )

    status = json.loads((out_dir / "daily_update_status.json").read_text())
    assert status["latest_calibration_date"] is None
    assert status["calibration_artifacts_available"] is False
    assert "live daily snapshots" in status["calibration_warning"]
    assert not (out_dir / "calibration_summary.json").exists()
    assert not (fig_dir / "01_data_distribution.png").exists()
