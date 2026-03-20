from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_panel(path: str | Path) -> pd.DataFrame:
    panel_path = Path(path)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    suffix = panel_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(panel_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(panel_path)
    raise RuntimeError(f"Unsupported panel file extension: {panel_path.suffix}")


def write_panel(df: pd.DataFrame, path: str | Path) -> None:
    panel_path = Path(path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = panel_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(panel_path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(panel_path, index=False)
        return
    raise RuntimeError(f"Unsupported panel file extension: {panel_path.suffix}")


def dedupe_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df.copy()
    panel["date"] = pd.to_datetime(panel.get("date"), errors="coerce")

    if "contract_symbol" in panel.columns:
        subset = ["date", "contract_symbol"]
    elif "contractSymbol" in panel.columns:
        subset = ["date", "contractSymbol"]
    else:
        subset = ["date", "maturity_date", "option_type", "strike"]

    panel = panel.drop_duplicates(subset=subset)
    sort_cols = [c for c in ["date", "maturity_date", "option_type", "strike"] if c in panel.columns]
    if sort_cols:
        panel = panel.sort_values(sort_cols)
    return panel.reset_index(drop=True)


def append_panel_rows(panel_path: str | Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    path = Path(panel_path)
    if path.exists():
        existing = load_panel(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = dedupe_panel(combined)
    write_panel(combined, path)
    return combined


def build_panel_manifest(
    panel: pd.DataFrame,
    *,
    panel_path: str | Path,
    ticker: str,
    provider: str,
    snapshot_time_local: str,
    timezone: str,
    archive_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = panel.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    dates = out["date"].dropna().sort_values()

    manifest: dict[str, Any] = {
        "ticker": str(ticker).upper(),
        "provider": provider,
        "panel_file": str(Path(panel_path)),
        "snapshot_time_local": snapshot_time_local,
        "timezone": timezone,
        "rows": int(len(out)),
        "n_dates": int(dates.nunique()) if not dates.empty else 0,
        "n_contracts": int(out["contract_symbol"].nunique()) if "contract_symbol" in out.columns else None,
        "date_start": dates.iloc[0].date().isoformat() if not dates.empty else None,
        "date_end": dates.iloc[-1].date().isoformat() if not dates.empty else None,
        "archive_root": str(Path(archive_root)) if archive_root is not None else None,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_panel_manifest(
    panel: pd.DataFrame,
    manifest_path: str | Path,
    *,
    panel_path: str | Path,
    ticker: str,
    provider: str,
    snapshot_time_local: str,
    timezone: str,
    archive_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_panel_manifest(
        panel,
        panel_path=panel_path,
        ticker=ticker,
        provider=provider,
        snapshot_time_local=snapshot_time_local,
        timezone=timezone,
        archive_root=archive_root,
        extra=extra,
    )
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
