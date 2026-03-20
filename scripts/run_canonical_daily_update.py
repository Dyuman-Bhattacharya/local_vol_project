#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.panel_store import load_panel, write_panel_manifest
from market_data.loaders import load_any


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must be a mapping: {path}")
    return data


def today_in_timezone(tz_name: str) -> str:
    return datetime.now(ZoneInfo(tz_name)).date().isoformat()


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def latest_snapshot_path(*, archive_root: Path, ticker: str, run_date: str, panel_manifest: Path, panel_file: Path) -> tuple[str, Path | None]:
    direct = archive_root / ticker.upper() / str(run_date) / "option_chain_snapshot.parquet"
    if direct.exists():
        return str(run_date), direct

    latest_date: str | None = None
    if panel_manifest.exists():
        try:
            payload = json.loads(panel_manifest.read_text(encoding="utf-8"))
            latest_date = payload.get("latest_collection_date") or payload.get("date_end")
        except Exception:
            latest_date = None
    if latest_date is None and panel_file.exists():
        panel_df = load_panel(panel_file)
        if not panel_df.empty:
            latest_date = str(pd.to_datetime(panel_df["date"]).max().date())

    if latest_date is None:
        return str(run_date), None

    path = archive_root / ticker.upper() / str(latest_date) / "option_chain_snapshot.parquet"
    return str(latest_date), path if path.exists() else None


def candidate_snapshot_paths(*, archive_root: Path, ticker: str, run_date: str, panel_manifest: Path, panel_file: Path) -> list[tuple[str, Path]]:
    candidates: list[str] = []
    if run_date:
        candidates.append(str(run_date))
    if panel_manifest.exists():
        try:
            payload = json.loads(panel_manifest.read_text(encoding="utf-8"))
            for key in ["latest_collection_date", "date_end", "date_start"]:
                val = payload.get(key)
                if val:
                    candidates.append(str(val))
        except Exception:
            pass
    if panel_file.exists():
        panel_df = load_panel(panel_file)
        if not panel_df.empty and "date" in panel_df.columns:
            dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna().dt.date.astype(str).unique().tolist()
            candidates.extend(sorted(dates, reverse=True))

    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for d in candidates:
        if d in seen:
            continue
        seen.add(d)
        path = archive_root / ticker.upper() / d / "option_chain_snapshot.parquet"
        if path.exists():
            out.append((d, path))
    return out


def clear_daily_calibration_outputs(out_dir: Path) -> None:
    stale_files = [
        "cleaned_options_data.csv",
        "chain.parquet",
        "surface_slice_summary.csv",
        "call_price_grid.npz",
        "arbitrage_free_iv_surface.npz",
        "arbitrage_free_iv_surface.pkl",
        "density.npz",
        "local_vol_grid.npz",
        "local_vol_surface.npz",
        "local_vol_surface.pkl",
        "arbitrage_diagnostics.json",
        "calibration_summary.json",
        "repricing_validation.csv",
    ]
    for name in stale_files:
        path = out_dir / name
        if path.exists():
            path.unlink()

    fig_dir = out_dir / "figures"
    stale_figs = [
        "01_data_distribution.png",
        "01_iv_surface_market.png",
        "01_outliers.png",
        "02_surface_fit_quality.png",
        "02_arbitrage_checks.png",
        "02_call_price_grid.png",
        "02_density_validation.png",
        "03_local_vol_surface.png",
        "03_local_vol_slices.png",
        "04_lv_diagnostics.png",
        "04_regularization_impact.png",
        "05_repricing_errors.png",
        "05_repricing_by_moneyness.png",
        "05_repricing_by_maturity.png",
    ]
    for name in stale_figs:
        path = fig_dir / name
        if path.exists():
            path.unlink()


def archive_canonical_run(*, out_dir: Path, run_date: str, panel_manifest: Path, status: dict[str, Any]) -> None:
    archive_root = out_dir / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    panel_info: dict[str, Any] = {}
    if panel_manifest.exists():
        try:
            panel_info = json.loads(panel_manifest.read_text(encoding="utf-8"))
        except Exception:
            panel_info = {}
    n_dates = int(panel_info.get("n_dates", 0) or 0)
    panel_end = str(panel_info.get("date_end", run_date))
    run_key = f"{panel_end}_n{n_dates:02d}"
    run_dir = archive_root / run_key
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "report.md",
        "daily_update_status.json",
        "daily_market_backtest_summary.json",
        "daily_market_backtest_BS.csv",
        "daily_market_backtest_LocalVol.csv",
        "selected_contracts.csv",
        "calibration_summary.json",
        "repricing_validation.csv",
        "surface_slice_summary.csv",
        "arbitrage_diagnostics.json",
    ]
    dirs_to_copy = ["figures"]

    copied_files: list[str] = []
    copied_dirs: list[str] = []
    for name in files_to_copy:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)
            copied_files.append(name)
    for name in dirs_to_copy:
        src = out_dir / name
        dst = run_dir / name
        if src.exists():
            shutil.copytree(src, dst)
            copied_dirs.append(name)

    milestone_tags = []
    if n_dates >= 6:
        milestone_tags.append("post_minimum")
    if n_dates in {6, 10}:
        milestone_tags.append(f"n{n_dates}")

    entry = {
        "run_key": run_key,
        "run_date": run_date,
        "panel_date_end": panel_end,
        "n_panel_dates": n_dates,
        "status": status.get("calibration_warning"),
        "calibration_artifacts_available": bool(status.get("calibration_artifacts_available", False)),
        "latest_calibration_date": status.get("latest_calibration_date"),
        "archive_dir": str(run_dir),
        "copied_files": copied_files,
        "copied_dirs": copied_dirs,
        "milestone_tags": milestone_tags,
    }

    index_path = archive_root / "archive_index.json"
    entries: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    entries = [e for e in entries if e.get("run_key") != run_key]
    entries.append(entry)
    entries = sorted(entries, key=lambda e: (str(e.get("panel_date_end", "")), int(e.get("n_panel_dates", 0))))
    index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    snapshot_summary = {
        "run_key": run_key,
        "run_date": run_date,
        "panel_date_end": panel_end,
        "n_panel_dates": n_dates,
        "milestone_tags": milestone_tags,
        "archive_index": str(index_path),
        "archive_dir": str(run_dir),
    }
    (out_dir / "archive_status.json").write_text(json.dumps(snapshot_summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Canonical daily update: append the 15:45 ET SPY snapshot, then refresh the short-dated hedge study and report."
    )
    ap.add_argument("--config", default="config/daily_collection.yaml")
    ap.add_argument("--date", default=None, help="Override collection date YYYY-MM-DD; default is today in config timezone")
    ap.add_argument("--skip_collection", action="store_true", help="Reuse the existing panel without fetching/appending a new snapshot")
    ap.add_argument("--input_file", default=None, help="Optional local snapshot file to archive instead of fetching live")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = PROJECT_ROOT / args.config
    cfg = load_config(cfg_path)

    collection = cfg.get("collection", {})
    backtest = cfg.get("backtest", {})
    report = cfg.get("report", {})

    timezone = str(collection.get("timezone", "America/New_York"))
    provider = str(collection.get("provider", "yfinance"))
    run_date = args.date or today_in_timezone(timezone)
    py = sys.executable

    panel_file = PROJECT_ROOT / str(collection.get("panel_out", "data/processed/spy_options_panel.parquet"))
    panel_manifest = PROJECT_ROOT / str(collection.get("panel_manifest", "data/processed/spy_options_panel_manifest.json"))
    archive_root = PROJECT_ROOT / str(collection.get("archive_root", "data/archive"))

    if not args.skip_collection:
        collect_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "archive_option_snapshot.py"),
            "--provider",
            provider,
            "--ticker",
            str(collection.get("ticker", "SPY")),
            "--date",
            run_date,
            "--archive_root",
            str(archive_root),
            "--panel_out",
            str(panel_file),
            "--panel_manifest",
            str(panel_manifest),
            "--timezone",
            timezone,
            "--snapshot_time",
            str(collection.get("snapshot_time", "15:45")),
            "--min_days",
            str(int(collection.get("min_days", 1))),
            "--max_days",
            str(int(collection.get("max_days", 365))),
        ]
        max_expiries = collection.get("max_expiries")
        if max_expiries is not None:
            collect_cmd += ["--max_expiries", str(int(max_expiries))]
        risk_free_rate = collection.get("risk_free_rate")
        if risk_free_rate is not None:
            collect_cmd += ["--risk_free_rate", str(float(risk_free_rate))]
        dividend_yield = collection.get("dividend_yield")
        if dividend_yield is not None:
            collect_cmd += ["--dividend_yield", str(float(dividend_yield))]
        if args.input_file:
            collect_cmd += ["--input_file", args.input_file]
        run_cmd(collect_cmd)

    if not panel_file.exists():
        raise RuntimeError(f"Canonical panel file does not exist: {panel_file}")
    if not panel_manifest.exists():
        panel_df = load_panel(panel_file)
        write_panel_manifest(
            panel_df,
            panel_manifest,
            panel_path=panel_file,
            ticker=str(collection.get("ticker", "SPY")),
            provider=f"{provider}_snapshot_archive",
            snapshot_time_local=str(collection.get("snapshot_time", "15:45")),
            timezone=timezone,
            archive_root=archive_root,
            extra={"collector_mode": "existing_panel"},
        )

    out_dir = PROJECT_ROOT / str(backtest.get("output_dir", "output/canonical_daily"))
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_daily_calibration_outputs(out_dir)

    calib_date = None
    calib_snapshot = None
    calibration_warning = None
    for cand_date, cand_snapshot in candidate_snapshot_paths(
        archive_root=archive_root,
        ticker=str(collection.get("ticker", "SPY")),
        run_date=run_date,
        panel_manifest=panel_manifest,
        panel_file=panel_file,
    ):
        calibration_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "run_snapshot_calibration.py"),
            "--ticker",
            str(collection.get("ticker", "SPY")),
            "--date",
            str(cand_date),
            "--out",
            str(out_dir),
            "--input_file",
            str(cand_snapshot),
            "--option_type",
            str(backtest.get("option_type", "call")),
            "--min_volume",
            str(float(backtest.get("min_volume", 10.0))),
            "--min_open_interest",
            str(float(backtest.get("min_open_interest", 50.0))),
            "--moneyness_min",
            str(float(backtest.get("moneyness_min", 0.82))),
            "--moneyness_max",
            str(float(backtest.get("moneyness_max", 1.18))),
            "--max_relative_spread",
            str(float(backtest.get("max_relative_spread", 0.08))),
            "--iv_min",
            str(float(backtest.get("iv_min", 0.03))),
            "--iv_max",
            str(float(backtest.get("iv_max", 1.50))),
            "--min_points_per_slice",
            str(int(backtest.get("min_points_per_slice", 8))),
            "--n_strikes",
            str(int(backtest.get("n_strikes", 60))),
        ]
        try:
            run_cmd(calibration_cmd)
            run_cmd(
                [
                    py,
                    str(PROJECT_ROOT / "scripts" / "run_pricing_validation.py"),
                    "--calib_dir",
                    str(out_dir),
                ]
            )
            calib_date = cand_date
            calib_snapshot = cand_snapshot
            break
        except subprocess.CalledProcessError as e:
            calibration_warning = f"Calibration failed for snapshot {cand_date}; trying older snapshot."
            print(calibration_warning, flush=True)
            continue

    if calib_date is None:
        calibration_warning = "Calibration failed for all available live daily snapshots."

    if bool(backtest.get("enabled", True)):
        backtest_cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "run_backtest_market_panel.py"),
            "--ticker",
            str(collection.get("ticker", "SPY")),
            "--panel_file",
            str(panel_file),
            "--out",
            str(out_dir),
            "--option_type",
            str(backtest.get("option_type", "call")),
            "--strike_mode",
            str(backtest.get("strike_mode", "atm")),
            "--T",
            str(float(backtest.get("T_years", 8.0 / 365.0))),
            "--entry_step_days",
            str(int(backtest.get("entry_step_days", 1))),
            "--contract_moneyness_band",
            str(float(backtest.get("contract_moneyness_band", 0.02))),
            "--contract_max_relative_spread",
            str(float(backtest.get("contract_max_relative_spread", 0.05))),
            "--contract_min_volume",
            str(float(backtest.get("contract_min_volume", 25.0))),
            "--contract_min_open_interest",
            str(float(backtest.get("contract_min_open_interest", 200.0))),
            "--tx_bps",
            str(float(backtest.get("tx_bps", 10.0))),
            "--pde_space",
            str(int(backtest.get("pde_space", 120))),
            "--pde_time",
            str(int(backtest.get("pde_time", 100))),
            "--min_volume",
            str(float(backtest.get("min_volume", 10.0))),
            "--min_open_interest",
            str(float(backtest.get("min_open_interest", 50.0))),
            "--moneyness_min",
            str(float(backtest.get("moneyness_min", 0.82))),
            "--moneyness_max",
            str(float(backtest.get("moneyness_max", 1.18))),
            "--max_relative_spread",
            str(float(backtest.get("max_relative_spread", 0.08))),
            "--iv_min",
            str(float(backtest.get("iv_min", 0.03))),
            "--iv_max",
            str(float(backtest.get("iv_max", 1.50))),
            "--min_points_per_slice",
            str(int(backtest.get("min_points_per_slice", 8))),
            "--n_strikes",
            str(int(backtest.get("n_strikes", 60))),
            "--min_panel_dates",
            str(int(backtest.get("min_panel_dates", 6))),
        ]
        if backtest.get("max_contracts") is not None:
            backtest_cmd += ["--max_contracts", str(int(backtest["max_contracts"]))]
        if backtest.get("K_fixed") is not None:
            backtest_cmd += ["--K_fixed", str(float(backtest["K_fixed"]))]
        if backtest.get("K_mult") is not None:
            backtest_cmd += ["--K_mult", str(float(backtest["K_mult"]))]
        if bool(backtest.get("save_daily_calibrations", False)):
            backtest_cmd.append("--save_daily_calibrations")
        run_cmd(backtest_cmd)

    if bool(report.get("generate", True)):
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "scripts" / "generate_report.py"),
                "--output-dir",
                str(out_dir),
            ]
        )

    status = {
        "config": str(cfg_path),
        "provider": provider,
        "run_date": run_date,
        "timezone": timezone,
        "collection_performed": bool(not args.skip_collection),
        "panel_file": str(panel_file),
        "panel_manifest": str(panel_manifest),
        "report_dir": str(out_dir),
        "latest_calibration_date": calib_date,
        "latest_calibration_snapshot": str(calib_snapshot) if calib_snapshot is not None else None,
        "calibration_artifacts_available": bool(calib_date is not None),
        "calibration_warning": calibration_warning,
        "canonical_target_T_years": float(backtest.get("T_years", 8.0 / 365.0)),
        "canonical_snapshot_time_local": str(collection.get("snapshot_time", "15:45")),
    }
    (out_dir / "daily_update_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    archive_canonical_run(out_dir=out_dir, run_date=run_date, panel_manifest=panel_manifest, status=status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
