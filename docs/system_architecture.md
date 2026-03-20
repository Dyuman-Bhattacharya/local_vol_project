# System Architecture

The repository is easiest to understand as five connected stages:

1. collect and archive listed SPY option snapshots,
2. build an arbitrage-free call surface from the liquid short-dated core,
3. derive Dupire local volatility,
4. price with PDE and Monte Carlo,
5. evaluate BS and Local Vol on a dated panel of listed contracts.

![Architecture](assets/architecture.svg)

## Canonical Object Of Record

The raw interpolated total-variance surface is not treated as the final answer. The canonical object is the repaired call-price grid after static-arbitrage projection.

That repaired grid is then used to:

- derive an arbitrage-free IV surface,
- inspect risk-neutral density,
- extract Dupire local volatility,
- price with PDE and Monte Carlo,
- and support daily recalibrated hedging.

## Surface Construction

The calibration path uses a direct total-variance surface in strike and maturity. OTM puts and OTM calls are merged into a synthetic call dataset through put-call parity so both wings contribute information. The raw surface is converted into a dense call-price grid and projected so the exported output satisfies:

- monotonicity in strike,
- convexity in strike,
- monotonicity in maturity.

## Local Volatility

Dupire local volatility is extracted from the repaired call surface rather than from the raw fit. The local-vol stage includes regularization and a domain classification:

- trusted interior region,
- weakly supported boundary region,
- extrapolation or unsupported region.

That split matters because the repo only treats the trusted region as a serious repricing and diagnostics domain.

## Daily Panel And Hedge Study

Without a dated option panel, the project can calibrate and price on one date, but it cannot run the strongest real-data hedge study. The daily panel solves that by storing one full listed chain per date. The canonical hedge engine then:

- selects fixed listed contracts at entry,
- uses the observed market premium at entry,
- recalibrates the IV surface on each trade date,
- derives a fresh local-vol surface on each trade date,
- compares BS and Local Vol deltas with transaction costs.

## Main Code Paths

- [Calibration pipeline](../src/pipeline/calibration.py)
- [Arbitrage projection](../src/arbitrage/projection.py)
- [Dupire extraction](../src/local_volatility/dupire.py)
- [PDE pricing](../src/pricing/pde_solver.py)
- [Market-panel hedging](../src/hedging/market_panel_backtest.py)
