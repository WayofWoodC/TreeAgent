# PE Quarterly Pipeline

A command-line research pipeline for quarterly PE nowcasting, model tuning, training, backtesting, and latest-signal generation.

The repository builds a quarterly panel for an S&P 500-style universe, trains an XGBoost model on one of two targets, and evaluates the resulting signal in a long-short backtest.

## Status

This README reflects the current code in this repository.

Verified on May 12, 2026:
- `pip install -e .` succeeds from the repository root
- the CLI entry point `pe-pipeline` is created correctly

## Environment Setup

### Python

Current package metadata requires:
- Python `>=3.10`

A Python 3.12 environment has already been verified for editable install.

### Install

From the repository root:

```bash
cd /Users/wood.chen/Desktop/Research/TreeAgent
python -m pip install -e .
```

### API Keys

The repository expects three environment variables in `.env` or in the shell:

```bash
FINNHUB_API_KEY=
FMP_API_KEY=
FRED_API_KEY=
```

You can create a local `.env` from `.env.example` and fill in the values.

Example shell setup:

```bash
export FINNHUB_API_KEY="your_finnhub_key"
export FMP_API_KEY="your_fmp_key"
export FRED_API_KEY="your_fred_key"
```

Important notes:
- `FINNHUB_API_KEY` is required for quarterly PE and fundamentals fetches.
- `FRED_API_KEY` is recommended for macro fetches. If it is not set, the pipeline falls back to the FRED graph endpoint.
- `FMP_API_KEY` is still present in config and `.env.example`, but the current fetch pipeline does not use FMP endpoints.

### Example run: 200-stock direct PE workflow with 60-day holding period

```bash
# no need to export if add keys in .env
export FINNHUB_API_KEY="your_finnhub_key"
export FRED_API_KEY="your_fred_key"
export FMP_API_KEY="your_fmp_key"

pe-pipeline fetch-data --limit 200
pe-pipeline build-dataset --target-name target_pe_current --start-year 2010
pe-pipeline tune --target-name target_pe_current --metric rmse --n-trials 25
pe-pipeline train --target-name target_pe_current
pe-pipeline backtest --top-n 5 --periods 60
```

## What The Pipeline Does

The pipeline has five main stages:

1. `fetch-data`
- Fetches universe, daily prices, quarterly PE ratios, quarterly fundamentals, and macro series.

2. `build-dataset`
- Builds the quarterly modeling dataset.
- Applies optional `--start-year` filtering.

3. `tune`
- Runs walk-forward hyperparameter tuning on the development set.

4. `train`
- Trains the final XGBoost model.
- Writes model artifacts, predictions, metrics, and feature importance.

5. `backtest`
- Reads the latest trained model metadata and the latest test predictions.
- Chooses the signal logic automatically based on the trained target.

There is also:
- `predict-latest` for latest-quarter ranked predictions
- `baseline` for a `pe_lag_1q` baseline workflow
- `run-all` for fetch -> build-dataset -> train -> backtest -> predict-latest

## Data Providers

Current live sources in code are:

- Universe:
  - Wikipedia S&P 500 constituents table
  - with local fallback logic if the live request fails

- Daily prices:
  - Yahoo Finance chart endpoint

- Quarterly PE ratios:
  - Finnhub `stock/metric`

- Quarterly fundamentals:
  - Finnhub `stock/metric`

- Macro:
  - FRED official observations API when `FRED_API_KEY` is available
  - fallback to the FRED graph CSV endpoint
  - fallback again to existing local raw macro files if all live fetches fail

## Repository Layout

```text
TreeAgent/
├── baseline/
├── data/
│   ├── raw/
│   ├── intermediate/
│   ├── processed/
│   └── predictions/
├── outputs/
│   ├── backtests/
│   ├── models/
│   └── reports/
├── pe_pipeline/
│   ├── backtesting/
│   ├── data_sources/
│   ├── features/
│   ├── modeling/
│   ├── pipelines/
│   └── utils/
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

### Directory Meanings

- `pe_pipeline/`
  - package source code

- `data/raw/`
  - raw fetched source data in CSV and Parquet

- `data/intermediate/`
  - intermediate feature tables and latest merged feature snapshot

- `data/processed/`
  - final quarterly modeling dataset

- `data/predictions/`
  - latest ranked predictions written by `predict-latest`

- `outputs/models/`
  - trained XGBoost model, preprocessor, model metadata

- `outputs/reports/`
  - training metrics, validation/test predictions, feature importance, tuning results, data quality report

- `outputs/backtests/`
  - backtest return tables, portfolio summary/stats, return plot, equity curve plot

- `baseline/`
  - baseline code and baseline-specific outputs

## CLI Overview

Top-level help:

```bash
pe-pipeline --help
```

Available commands:

- `fetch-data`
- `build-dataset`
- `train`
- `tune`
- `backtest`
- `baseline`
- `predict-latest`
- `run-all`

## Command Reference

### 1. `fetch-data`

```bash
pe-pipeline fetch-data [--limit N]
```

Arguments:
- `--limit N`
  - Optional.
  - Restricts the fetched universe to the first `N` tickers after universe construction.
  - Useful for smoke tests.

Writes:
- `data/raw/universe.*`
- `data/raw/prices/daily_prices.*`
- `data/raw/pe_ratios/quarterly_pe_ratios.*`
- `data/raw/fundamentals/quarterly_fundamentals.*`
- `data/raw/macro/macro_series.*`

### 2. `build-dataset`

```bash
pe-pipeline build-dataset --target-name {target_pe_current,target_pe_scaled_change} [--start-year YYYY]
```

Arguments:
- `--target-name`
  - `target_pe_current`
  - `target_pe_scaled_change`
- `--start-year YYYY`
  - Optional.
  - Keeps only rows with `quarter_end >= YYYY-01-01` after the dataset is built.

Writes:
- `data/processed/modeling_dataset.parquet`
- `outputs/reports/data_quality_report.json`

### 3. `tune`

```bash
pe-pipeline tune --target-name {target_pe_current,target_pe_scaled_change} --metric {rmse,spearman} [--n-trials N]
```

Arguments:
- `--target-name`
  - Training target used during tuning.
- `--metric`
  - `rmse`: choose hyperparameters by lower walk-forward validation RMSE
  - `spearman`: choose hyperparameters by higher walk-forward validation rank correlation
- `--n-trials`
  - Number of Optuna trials

Behavior:
- Uses walk-forward CV on the development set.
- Leaves the final test block untouched.

Writes:
- `outputs/reports/best_params.json`
- `outputs/reports/tuning_results.csv`

### 4. `train`

```bash
pe-pipeline train --target-name {target_pe_current,target_pe_scaled_change}
```

Behavior:
- Loads `data/processed/modeling_dataset.parquet`
- Applies the fixed time split
- Automatically reads `outputs/reports/best_params.json` if it exists
- Trains the XGBoost model
- Saves validation and test predictions

Writes:
- `outputs/models/xgb_model.json`
- `outputs/models/preprocessor.joblib`
- `outputs/models/model_metadata.json`
- `outputs/reports/training_metrics.json`
- `outputs/reports/feature_importance.csv`
- `outputs/reports/validation_predictions.csv`
- `outputs/reports/test_predictions.csv`

### 5. `backtest`

```bash
pe-pipeline backtest [--top-n N] [--periods D]
```

Arguments:
- `--top-n N`
  - Number of names in the long basket and number of names in the short basket.
- `--periods D`
  - Optional holding period in trading days.
  - If omitted, uses the default next-quarter return.
  - If provided, uses the realized forward return from the first trading day after signal formation through `D` trading days later.

Important behavior:
- The signal logic is chosen automatically from the trained model metadata.
- `backtest` does not ask you to specify the target again.
- It reads `outputs/models/model_metadata.json` to determine how prediction should be converted into a signal.

Writes:
- `outputs/backtests/quarterly_returns.csv`
- `outputs/backtests/portfolio_summary.json`
- `outputs/backtests/portfolio_stats.csv`
- `outputs/backtests/quarterly_returns_plot.svg`
- `outputs/backtests/quarterly_returns_plot.png`
- `outputs/backtests/equity_curves_plot.svg`
- `outputs/backtests/equity_curves_plot.png`

### 6. `baseline`

```bash
pe-pipeline baseline --target-name {target_pe_current,target_pe_scaled_change} [--start-year YYYY] [--top-n N] [--periods D]
```

Behavior:
- Rebuilds the processed dataset using the requested target and optional start year
- Uses a simple `pe_lag_1q` baseline prediction rule
- Writes all baseline-specific outputs into the repository-level `baseline/` directory

Important note:
- For `target_pe_current`, the baseline prediction is exactly `pe_lag_1q`.
- The direct-PE trading signal is defined as relative change versus `pe_lag_1q`.
- Therefore this baseline can produce a flat zero signal and a zero backtest by construction.

### 7. `predict-latest`

```bash
pe-pipeline predict-latest
```

Behavior:
- Rebuilds the latest merged feature snapshot from raw data
- Loads the latest trained model and preprocessor
- Applies target-consistent signal logic
- Sorts by signal descending

Writes:
- `data/predictions/latest_predictions.csv`

### 8. `run-all`

```bash
pe-pipeline run-all --target-name {target_pe_current,target_pe_scaled_change} [--top-n N] [--limit N] [--start-year YYYY] [--periods D]
```

Runs:
1. `fetch-data`
2. `build-dataset`
3. `train`
4. `backtest`
5. `predict-latest`

Important note:
- `run-all` does not run `tune`.
- If you want tuned hyperparameters, run `tune` first and then `train`.

## Target Definitions

Only two public training targets are currently supported.

### `target_pe_current`

Definition:
- current-quarter PE ratio

Conceptually:
- `y_t = PE_t`

### `target_pe_scaled_change`

Definition:
- current-quarter PE change versus previous quarter, scaled by the magnitude of previous-quarter PE

Formula:

```text
target_pe_scaled_change = (PE_t - PE_{t-1}) / (abs(PE_{t-1}) + epsilon)
```

This is not plain percent change. The absolute value and epsilon are used to stabilize the target when lagged PE is negative or near zero.

## How Backtest Signal Is Chosen

This is important to avoid confusion.

The backtest reads `outputs/models/model_metadata.json` and uses the saved `target_name`.

### If the trained target is `target_pe_current`

The model predicts a PE level.

The backtest converts that into a trading signal using:

```text
signal = (prediction - pe_lag_1q) / abs(pe_lag_1q)
```

This makes the signal consistent with a relative change view instead of ranking directly on raw PE level.

### If the trained target is `target_pe_scaled_change`

The model prediction is already the processed signal target.

The backtest uses:

```text
signal = prediction
```

### Ranking Rule

For each quarter:
- long the top `N` signals
- short the bottom `N` signals
- equal weight within long and within short

### Long-Short Return Rule

For each backtest period:
- `long_return` = average realized return of the long basket
- `short_asset_return` = average realized return of the short basket itself
- `short_return = - short_asset_return`
- `long_short_return = 0.5 * long_return + 0.5 * short_return`

Interpretation:
- total capital is 1
- 0.5 allocated to the long leg
- 0.5 allocated to the short leg
- net exposure is approximately 0

## Holding Period Logic

Default behavior:
- The backtest uses `next_quarter_return`.

With `--periods D`:
- the signal is still generated once per quarter
- the position is opened on the first real trading day after quarter end
- the position is closed `D` trading days later
- the portfolio is then assumed to stay in cash until the next quarterly signal

Important annualization rule:
- Even when `--periods D` is used, the signal frequency remains quarterly.
- Therefore annualized return, annualized volatility, Sharpe, and Sortino are annualized using `4` periods per year, not `252 / D`.

## Recommended Workflows


### Example run: 200-stock direct PE workflow with 60-day holding period

```bash
cd /Users/wood.chen/Desktop/Research/TreeAgent
export FINNHUB_API_KEY="your_finnhub_key"
export FRED_API_KEY="your_fred_key"
export FMP_API_KEY="your_fmp_key"

pe-pipeline fetch-data --limit 200
pe-pipeline build-dataset --target-name target_pe_current --start-year 2010
pe-pipeline tune --target-name target_pe_current --metric rmse --n-trials 25
pe-pipeline train --target-name target_pe_current
pe-pipeline backtest --top-n 5 --periods 60
```

### Same workflow with a 20-trading-day holding period

```bash
pe-pipeline backtest --top-n 3 --periods 20
```

### Full workflow for scaled-change target

```bash
cd /Users/wood.chen/Desktop/Research/TreeAgent
export FINNHUB_API_KEY="your_finnhub_key"
export FRED_API_KEY="your_fred_key"
export FMP_API_KEY="your_fmp_key"

pe-pipeline fetch-data --limit 20
pe-pipeline build-dataset --target-name target_pe_scaled_change --start-year 2010
pe-pipeline tune --target-name target_pe_scaled_change --metric spearman --n-trials 25
pe-pipeline train --target-name target_pe_scaled_change
pe-pipeline backtest --top-n 3
pe-pipeline predict-latest
```

### If raw data is already present

Direct PE target:

```bash
pe-pipeline build-dataset --target-name target_pe_current --start-year 2010
pe-pipeline tune --target-name target_pe_current --metric rmse --n-trials 25
pe-pipeline train --target-name target_pe_current
pe-pipeline backtest --top-n 3
```

Scaled-change target:

```bash
pe-pipeline build-dataset --target-name target_pe_scaled_change --start-year 2010
pe-pipeline tune --target-name target_pe_scaled_change --metric spearman --n-trials 25
pe-pipeline train --target-name target_pe_scaled_change
pe-pipeline backtest --top-n 3
```

## Main Output Files

### Training and tuning

- `outputs/reports/best_params.json`
- `outputs/reports/tuning_results.csv`
- `outputs/reports/training_metrics.json`
- `outputs/reports/feature_importance.csv`
- `outputs/reports/validation_predictions.csv`
- `outputs/reports/test_predictions.csv`
- `outputs/models/model_metadata.json`

### Backtest

- `outputs/backtests/quarterly_returns.csv`
- `outputs/backtests/portfolio_summary.json`
- `outputs/backtests/portfolio_stats.csv`
- `outputs/backtests/quarterly_returns_plot.svg`
- `outputs/backtests/equity_curves_plot.svg`

### Latest prediction

- `data/predictions/latest_predictions.csv`

### Baseline

- `baseline/baseline_metrics.json`
- `baseline/test_predictions.csv`
- `baseline/quarterly_returns.csv`
- `baseline/portfolio_summary.json`
- `baseline/portfolio_stats.csv`

## Notes On Reproducibility

- `pip install -e .` has been verified in this repository.
- The current repo depends on live external data providers for `fetch-data`.
- If live macro fetches fail, the code can reuse an existing local raw macro dataset.
- `run-all` intentionally skips tuning; use `tune` explicitly when you want model selection before final training.

## Minimal Sanity Checks

After installation:

```bash
pe-pipeline --help
pe-pipeline build-dataset --help
pe-pipeline tune --help
pe-pipeline backtest --help
pe-pipeline baseline --help
```

These commands have already been verified from the current repository checkout.
