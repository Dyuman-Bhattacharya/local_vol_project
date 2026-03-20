# Findings

This page separates the repository's two result modes:

1. a static reference artifact set from the earlier single-snapshot workflow,
2. the canonical daily dated-panel workflow built on Theta option-chain data.

## Reference Snapshot Workflow

The reference artifact set shows the full calibration and pricing pipeline on one completed snapshot-based run. It is useful for understanding the shape of the outputs and for viewing stable figures that do not depend on the latest local `output/` directory.

- [Reference artifact report](reference_outputs/reference_report.md)

Representative figures:

![Projected call grid](assets/surface_call_grid.png)

![Local-vol surface](assets/local_vol_surface.png)

![Hedge-error decomposition](assets/hedging_error_decomposition.png)

## Canonical Daily Workflow

The canonical real-data workflow is the daily dated-panel path:

- [daily update script](../scripts/run_canonical_daily_update.py)
- [daily collection config](../config/daily_collection.yaml)
- generated locally under `output/canonical_daily/report.md`

This workflow should be read as a Theta-based liquid-core study:

- short-dated SPY,
- near-ATM contracts,
- tighter spread and liquidity filters,
- and daily recalibration on a dated option panel.

As of the latest local run, the canonical panel contains 21 dated snapshots from February 18, 2026 through March 18, 2026, and the active hedge study spans 10 fixed near-ATM listed contracts.

## Current Daily Metrics

| Metric | Value |
| --- | ---: |
| Raw static-arbitrage failures before projection | `140` |
| Static-arbitrage failures after projection | `0` |
| Weighted fit inside bid/ask | `10.08%` |
| Trusted-domain repricing MAE | `2.9125` |
| Projection RMSE adjustment | `0.1338` |
| Density-mass minimum | `0.3045` |
| Trusted local-vol domain fraction | `12.56%` |
| Supported local-vol domain fraction | `88.22%` |
| BS pricing RMSE | `0.1888` |
| Local Vol pricing RMSE | `1.4566` |
| BS replication net RMSE | `2.2176` |
| Local Vol replication net RMSE | `2.6675` |
| BS net market-PnL RMSE | `2.1460` |
| Local Vol net market-PnL RMSE | `1.7217` |
| BS mean absolute MTM error | `1.6382` |
| Local Vol mean absolute MTM error | `0.6845` |

Verification:

- full suite status: `81 passed`

## Interpretation

The current result is mixed rather than one-sided.

- Black-Scholes is better on entry pricing and replication RMSE.
- Local Vol is better on mark-to-market error and net market-PnL RMSE.

That split is useful. It shows the repo is not collapsing the entire comparison into one headline number, and it makes the tradeoff between calibration quality and downstream hedge behavior visible.

## Related Reading

- [Project Summary](project_summary.md)
- [System Architecture](system_architecture.md)
- [Reference Outputs](reference_outputs/README.md)
