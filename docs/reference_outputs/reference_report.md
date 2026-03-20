# Reference Artifact Report

This page summarizes the completed historical single-snapshot artifact set that is kept with the repository as a stable reference. These numbers are intentionally preserved as a reference point; they are not the current daily-panel metrics.

## Outcome

| Metric | Value |
| --- | ---: |
| Raw static-arbitrage failures before projection | `25` |
| Final static-arbitrage failures after projection | `0` |
| Projection RMSE adjustment | `0.0173061` |
| Local-vol cap-bound cells | `0%` |
| Mean absolute repricing error | `0.956877` |
| Mean absolute BS hedge error | `1.38201` |
| Mean absolute Local Vol hedge error | `5.03276` |

This artifact set is useful because it preserves a complete, inspectable chain of outputs:

1. cleaned listed option data,
2. an arbitrage-repaired call surface,
3. a Dupire local-vol surface,
4. pricing diagnostics,
5. hedge-study diagnostics.

## Representative Figures

Projected arbitrage-repaired call surface:

![Projected call grid](../assets/surface_call_grid.png)

Regularized Dupire local-vol surface:

![Local vol surface](../assets/local_vol_surface.png)

Hedge-study error decomposition:

![Hedging error decomposition](../assets/hedging_error_decomposition.png)

## Relation To The Daily Workflow

This reference artifact set belongs to the earlier single-snapshot workflow. The canonical architecture for the repository is now the daily dated-panel workflow documented in:

- [Findings](../findings.md)
- [System Architecture](../system_architecture.md)
- [README](../../README.md)
