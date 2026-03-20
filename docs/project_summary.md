# Project Summary

This repository builds a daily options-modeling workflow around listed SPY option chains. It collects one full option snapshot per trading day, constructs an arbitrage-free surface, derives Dupire local volatility, prices with PDE and Monte Carlo, and evaluates Black-Scholes versus Local Vol hedging with transaction costs on fixed listed contracts.

The canonical workflow is intentionally centered on the liquid short-dated SPY core. Option-chain data comes from Theta, while a small Yahoo fallback is retained for SPY spot and carry inputs.

## Pipeline Snapshot

| Area | What the repository does |
| --- | --- |
| Collection | Archives a full SPY option snapshot each trading day at `15:45` America/New_York |
| Surface construction | Fits an interpolated total-variance surface and projects the exported call grid onto the static no-arbitrage set |
| Arbitrage constraints | Enforces calendar monotonicity and butterfly convexity on the exported grid |
| Local volatility | Extracts and regularizes Dupire local volatility from the repaired call surface |
| Pricing | Prices options with Crank-Nicolson PDE and Monte Carlo |
| Hedging | Compares BS and Local Vol delta hedging with explicit transaction costs on fixed near-ATM listed contracts |

## Architecture

![Architecture](assets/architecture.svg)

## Representative Figures

Projected arbitrage-repaired call surface:

![Projected call grid](assets/surface_call_grid.png)

Regularized Dupire local-vol surface:

![Local vol surface](assets/local_vol_surface.png)

Hedge-study error decomposition:

![Hedging error decomposition](assets/hedging_error_decomposition.png)

## Current Benchmarks

The table below contrasts the static single-snapshot reference artifact set with the current canonical daily workflow.

| Metric | Reference snapshot workflow | Canonical daily workflow |
| --- | ---: | ---: |
| Static-arbitrage failures after projection | `0` | `0` |
| Weighted fit inside bid/ask | `n/a` | `10.08%` |
| Mean absolute repricing error | `0.9569` | `2.9125` |
| Raw static-arbitrage failures before projection | `25` | `140` |
| Projection RMSE adjustment | `0.0173` | `0.1338` |
| Density-mass minimum | `0.4026` | `0.3045` |
| Trusted local-vol domain fraction | `n/a` | `12.56%` |
| BS pricing RMSE | `n/a` | `0.1888` |
| Local Vol pricing RMSE | `n/a` | `1.4566` |
| BS net market-PnL RMSE | `n/a` | `2.1460` |
| Local Vol net market-PnL RMSE | `n/a` | `1.7217` |
| Transaction costs modeled | `Yes` | `Yes` |
| Fixed listed contracts | `No` | `Yes` |
| Common market entry premium | `No` | `Yes` |
| Daily recalibration | `No` | `Yes` |

Verification:

- full suite status: `81 passed`

## Reading Path

- [README](../README.md)
- [System Architecture](system_architecture.md)
- [Data Workflow](data_workflow.md)
- [Findings](findings.md)
- [Reference Outputs](reference_outputs/README.md)
