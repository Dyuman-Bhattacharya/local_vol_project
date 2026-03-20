#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from market_data.loaders import load_any
from pipeline.calibration import (
    SnapshotCalibrationConfig,
    calibrate_option_snapshot,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a calibration sensitivity sweep over quote filters and local-vol regularization settings.")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--option_type", choices=["call", "put"], default="call")
    ap.add_argument("--n_strikes", type=int, default=60)
    return ap.parse_args()


def build_scenarios(base: SnapshotCalibrationConfig) -> list[tuple[str, SnapshotCalibrationConfig]]:
    return [
        ("baseline", base),
        ("tighter_spread", replace(base, max_relative_spread=0.04)),
        ("looser_spread", replace(base, max_relative_spread=0.08)),
        ("wider_moneyness", replace(base, moneyness_min=0.82, moneyness_max=1.18)),
        ("lower_open_interest", replace(base, min_open_interest=20.0)),
        ("higher_open_interest", replace(base, min_open_interest=100.0)),
        (
            "lighter_regularization",
            replace(
                base,
                regularization=replace(
                    base.regularization,
                    gaussian_sigma_T=0.25,
                    gaussian_sigma_K=0.40,
                    short_end_anchor_blend=0.10,
                    max_vol=2.5,
                ),
            ),
        ),
        (
            "stronger_regularization",
            replace(
                base,
                regularization=replace(
                    base.regularization,
                    gaussian_sigma_T=0.60,
                    gaussian_sigma_K=0.90,
                    short_end_anchor_blend=0.35,
                    max_vol=1.75,
                ),
            ),
        ),
        (
            "balanced_alt",
            replace(
                base,
                max_relative_spread=0.06,
                moneyness_min=0.83,
                moneyness_max=1.17,
                min_open_interest=25.0,
                regularization=replace(
                    base.regularization,
                    gaussian_sigma_T=0.35,
                    gaussian_sigma_K=0.55,
                    short_end_anchor_blend=0.25,
                    max_vol=1.85,
                ),
            ),
        ),
    ]


def score_row(row: dict) -> float:
    final_fails = float(row["final_static_arbitrage_fails"])
    density_mass_min = float(row["density_mass_min"])
    cap_frac = float(row["local_vol_cap_fraction"])
    proj_rmse = float(row["projection_rmse_adjustment"])
    local_vol_max = float(row["local_vol_max"])
    retained = float(row["n_after_filters"])
    density_penalty = max(0.0, 0.80 - density_mass_min)
    return (
        1000.0 * final_fails
        + 200.0 * density_penalty
        + 50.0 * cap_frac
        + 10.0 * proj_rmse
        + 0.5 * max(0.0, local_vol_max - 2.0)
        - 0.002 * retained
    )


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        rows.append(vals)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(vals) + " |" for vals in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_any(args.input_file)
    base = SnapshotCalibrationConfig(option_type=str(args.option_type), n_strikes=int(args.n_strikes))

    rows: list[dict] = []
    failures: list[dict] = []
    for name, cfg in build_scenarios(base):
        try:
            artifacts = calibrate_option_snapshot(
                df,
                ticker=str(args.ticker),
                valuation_date=str(args.date),
                cfg=cfg,
            )
            row = {
                "scenario": name,
                "n_after_filters": int(artifacts.summary["n_after_filters"]),
                "n_slices": int(artifacts.summary["n_slices"]),
                "raw_static_arbitrage_fails": int(artifacts.summary["arbitrage_counts_raw"]["n_fail"]),
                "final_static_arbitrage_fails": int(artifacts.summary["arbitrage_counts"]["n_fail"]),
                "projection_rmse_adjustment": float(artifacts.summary["projection_rmse_adjustment"]),
                "density_mass_min": float(artifacts.summary["density_mass_min"]),
                "density_mass_max": float(artifacts.summary["density_mass_max"]),
                "local_vol_max": float(artifacts.summary["local_vol_max"]),
                "local_vol_cap_fraction": float(artifacts.summary["local_vol_cap_fraction"]),
                "moneyness_min": float(cfg.moneyness_min),
                "moneyness_max": float(cfg.moneyness_max),
                "max_relative_spread": float(cfg.max_relative_spread),
                "min_open_interest": float(cfg.min_open_interest),
                "gaussian_sigma_T": float(cfg.regularization.gaussian_sigma_T),
                "gaussian_sigma_K": float(cfg.regularization.gaussian_sigma_K),
                "short_end_anchor_blend": float(cfg.regularization.short_end_anchor_blend),
                "regularization_max_vol": float(cfg.regularization.max_vol),
            }
            row["score"] = score_row(row)
            rows.append(row)
        except Exception as e:
            failures.append({"scenario": name, "error": str(e), "config": asdict(cfg)})

    if not rows:
        raise RuntimeError("All sensitivity scenarios failed.")

    result_df = pd.DataFrame(rows).sort_values(["score", "final_static_arbitrage_fails", "projection_rmse_adjustment"]).reset_index(drop=True)
    result_df.to_csv(out / "sensitivity_sweep.csv", index=False)
    (out / "sensitivity_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    best = result_df.iloc[0].to_dict()
    md_lines = [
        "# Calibration Sensitivity Sweep",
        "",
        f"Snapshot: `{args.input_file}`",
        f"Ticker/date: `{args.ticker}` / `{args.date}`",
        "",
        "## Best Scenario",
        "",
        f"- Scenario: `{best['scenario']}`",
        f"- Score: `{best['score']:.4f}`",
        f"- Final static-arbitrage fails: `{int(best['final_static_arbitrage_fails'])}`",
        f"- Density mass min: `{best['density_mass_min']:.4f}`",
        f"- Local-vol cap fraction: `{best['local_vol_cap_fraction']:.4f}`",
        f"- Projection RMSE adjustment: `{best['projection_rmse_adjustment']:.6f}`",
        "",
        "## Ranked Results",
        "",
        dataframe_to_markdown_table(result_df),
    ]
    (out / "calibration_sensitivity.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
