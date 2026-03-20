# Calibration Sensitivity Study

This page records a small calibration sweep on the local sample snapshot:

- snapshot: `data/raw/spy_options_snapshot.parquet`
- ticker/date: `SPY`, `2026-01-04`

The purpose was not to search a large hyperparameter space. It was to test whether nearby quote-filter and regularization choices materially changed the output surface.

## Main Outcome

On the older single-snapshot workflow, the strongest improvement came from widening the moneyness window:

- baseline at that time: `[0.85, 1.15]`
- best nearby setting on the sample snapshot: `[0.82, 1.18]`

That wider band:

- increased retained quotes from `681` to `723`,
- increased fitted maturity slices from `23` to `26`,
- improved `density_mass_min` from about `0.4026` to `0.4915`,
- still produced `0` final static-arbitrage failures after projection.

## Interpretation

This result belongs to the earlier snapshot-based configuration. It showed that the original single-snapshot filters were too narrow for that particular dataset.

The current Theta-based daily workflow now uses a different objective: fit a tighter liquid core more reliably rather than preserve as much strike coverage as possible. That is why the active daily configuration is intentionally narrower even though this reference study favored a wider band on the older snapshot workflow.

## Reproducing The Study

```powershell
poetry run python scripts\run_calibration_sensitivity_sweep.py `
  --input_file data\raw\spy_options_snapshot.parquet `
  --ticker SPY `
  --date 2026-01-04 `
  --out output\sensitivity_sweep
```
