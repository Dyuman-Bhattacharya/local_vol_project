# Data Workflow

## Daily Collection

The canonical workflow assumes one full SPY option snapshot per trading day at `15:45` America/New_York.

The collector:

- pulls the option chain from Theta,
- uses a small Yahoo fallback only for SPY spot and carry inputs,
- archives the raw snapshot under `data/archive/SPY/YYYY-MM-DD/`,
- appends it into `data/processed/spy_options_panel.parquet`,
- refreshes `data/processed/spy_options_panel_manifest.json`.

The panel can be both:

- backfilled from recent history with [backfill_theta_option_panel.py](../scripts/backfill_theta_option_panel.py),
- extended one day at a time by [run_canonical_daily_update.py](../scripts/run_canonical_daily_update.py).

## Validation And Cleaning

Cleaning happens during calibration and contract selection, not at collection time. The archive is preserved first, then the research pipeline filters the chain.

The main checks and filters include:

- valid option type,
- positive spot and strike,
- bid-ask sanity,
- computable nonnegative mids,
- minimum volume,
- minimum open interest,
- maximum relative spread,
- moneyness range,
- implied-volatility sanity range,
- minimum points per maturity slice.

The daily workflow intentionally narrows the option universe to a liquid short-dated core. The point is to fit a smaller trustworthy surface well rather than to fit the full chain badly.

## Panel Construction

Each archived snapshot is standardized into a common schema with fields such as:

- valuation date,
- maturity date,
- strike,
- option type,
- bid, ask, and midpoint,
- volume and open interest,
- spot, rate, and dividend inputs,
- contract symbol and vendor metadata.

That dated panel is what makes the historical hedge study possible. It allows the code to follow fixed listed contracts through time instead of replaying one frozen surface across arbitrary price paths.

## Daily Update Cycle

The daily update script performs the full operating cycle:

1. collect and archive the latest snapshot,
2. append it to the panel,
3. recalibrate the surface,
4. regenerate pricing diagnostics,
5. rerun the hedge study,
6. refresh the canonical report.

On Windows this is scheduled through [install_windows_daily_snapshot_task.py](../scripts/install_windows_daily_snapshot_task.py).
