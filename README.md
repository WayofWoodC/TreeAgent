# PE Quarterly Pipeline

Research repository for quarterly PE nowcasting, model training, backtesting, latest prediction export, and baseline comparisons.

This README is based on the current repository structure and current runnable entry points.

## Installation

### Python requirement

- Python `>= 3.10`

### Editable install

From the repository root:

```bash
cd TreeAgent
python -m pip install -e .
```

`pip install -e .` has already been verified for this repository.

## Environment Variables

The repository reads keys from the shell or from `.env`.

Current expected variables:

```bash
FINNHUB_API_KEY=
FMP_API_KEY=
FRED_API_KEY=
```

Example:

```bash
export FINNHUB_API_KEY="your_finnhub_key"
export FMP_API_KEY="your_fmp_key"
export FRED_API_KEY="your_fred_key"
```

Notes:

- `FINNHUB_API_KEY`
  - Required for quarterly PE ratios and quarterly fundamentals.

- `FRED_API_KEY`
  - Recommended for macro series fetches.
  - If missing, the code falls back to the FRED graph endpoint.

- `FMP_API_KEY`
  - Still present in config and `.env.example`.
  - The current live fetch pipeline does not rely on FMP for the main workflow.

## Typical Run Examples

### Main model: direct PE, 200 names, 60-day holding period

```bash
cd TreeAgent
export FINNHUB_API_KEY="your_finnhub_key"
export FRED_API_KEY="your_fred_key"
export FMP_API_KEY="your_fmp_key"

pe-pipeline fetch-data --limit 200
pe-pipeline build-dataset --target-name target_pe_current --start-year 2010
pe-pipeline tune --target-name target_pe_current --metric rmse --n-trials 25
pe-pipeline train --target-name target_pe_current
pe-pipeline backtest --top-n 5 --periods 60
```

### Classification baseline

```bash
python -m baseline.classification.run_classification --start-year 2010 --periods 60
```

### Lasso baseline

```bash
python -m baseline.Lasso.run_lasso --start-year 2010 --periods 60 --top-n 5
```

## Data Providers

Current providers used by the code:

- Universe
  - Wikipedia S&P 500 constituents table
  - with local fallback logic

- Daily prices
  - Yahoo Finance chart endpoint

- Quarterly PE ratios
  - Finnhub `stock/metric`

- Quarterly fundamentals
  - Finnhub `stock/metric`

- Macro
  - FRED API
  - fallback to FRED graph CSV
  - fallback to existing local macro raw files if all live requests fail

## Repository Structure

```text
TreeAgent/
├── baseline/
│   ├── prelag/
│   ├── classification/
│   └── Lasso/
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

### Main folders

- `pe_pipeline/`
  - Main package source code.
  - This is where the production pipeline logic lives.

- `pe_pipeline/data_sources/`
  - External data fetchers.
  - Universe, prices, PE ratios, fundamentals, macro.

- `pe_pipeline/features/`
  - Feature engineering and target construction.
  - Daily, macro, PE, fundamentals, dataset merge logic.

- `pe_pipeline/modeling/`
  - Feature selection, preprocessing, time split logic, training, tuning, model artifacts.

- `pe_pipeline/backtesting/`
  - Signal-to-portfolio conversion, forward return attachment, summary metrics, return/equity plots.

- `pe_pipeline/pipelines/`
  - Top-level workflows behind CLI commands such as `fetch-data`, `train`, `backtest`, and `run-all`.

- `data/raw/`
  - Raw fetched data.
  - CSV and Parquet.

- `data/intermediate/`
  - Intermediate feature tables and merged latest-quarter feature snapshots.

- `data/processed/`
  - Final modeling dataset used by `train` and `tune`.

- `data/predictions/`
  - Latest ranked prediction export from `predict-latest`.

- `outputs/models/`
  - Trained model files and metadata.

- `outputs/reports/`
  - Training metrics, feature importance, validation/test predictions, tuning results, data quality report.

- `outputs/backtests/`
  - Main model backtest results, quarterly return tables, summary stats, and plots.

- `baseline/prelag/`
  - Previous-quarter PE baseline outputs.

- `baseline/classification/`
  - 3-class baseline outputs.

- `baseline/Lasso/`
  - Lasso regression baseline outputs.

- `tests/`
  - Repository tests for data construction, splits, targets, and backtest logic.

## Main CLI Commands

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

### `fetch-data`

Fetch raw data into `data/raw/`.

```bash
pe-pipeline fetch-data [--limit N]
```

Optional parameters:

- `--limit N`
  - Restrict the fetched universe to the first `N` names.
  - Useful for smoke tests or smaller runs.

Writes:

- `data/raw/universe.*`
- `data/raw/prices/daily_prices.*`
- `data/raw/pe_ratios/quarterly_pe_ratios.*`
- `data/raw/fundamentals/quarterly_fundamentals.*`
- `data/raw/macro/macro_series.*`

### `build-dataset`

Build the final modeling dataset from the raw files.

```bash
pe-pipeline build-dataset --target-name {target_pe_current,target_pe_scaled_change} [--start-year YYYY]
```

Optional parameters:

- `--target-name`
  - `target_pe_current`
  - `target_pe_scaled_change`

- `--start-year YYYY`
  - Keep only rows with `quarter_end >= YYYY-01-01` after the dataset is built.

Writes:

- `data/processed/modeling_dataset.parquet`
- `outputs/reports/data_quality_report.json`

### `tune`

Run walk-forward tuning on the processed dataset.

```bash
pe-pipeline tune --target-name {target_pe_current,target_pe_scaled_change} --metric {rmse,spearman} [--n-trials N]
```

Optional parameters:

- `--target-name`
  - Training target used for tuning.

- `--metric`
  - `rmse`
  - `spearman`

- `--n-trials N`
  - Number of Optuna trials.
  - Default is `25`.

Writes:

- `outputs/reports/best_params.json`
- `outputs/reports/tuning_results.csv`

### `train`

Train the main XGBoost regression model.

```bash
pe-pipeline train --target-name {target_pe_current,target_pe_scaled_change}
```

Optional parameters:

- `--target-name`
  - Select the target used in model training.

Behavior:

- Reads `data/processed/modeling_dataset.parquet`
- Applies the fixed train / validation / test split
- Uses tuned hyperparameters automatically if `outputs/reports/best_params.json` exists

Writes:

- `outputs/models/xgb_model.json`
- `outputs/models/preprocessor.joblib`
- `outputs/models/model_metadata.json`
- `outputs/reports/training_metrics.json`
- `outputs/reports/feature_importance.csv`
- `outputs/reports/validation_predictions.csv`
- `outputs/reports/test_predictions.csv`

### `backtest`

Backtest the latest trained main model.

```bash
pe-pipeline backtest [--top-n N] [--periods D]
```

Optional parameters:

- `--top-n N`
  - Number of long names and number of short names per rebalance quarter.
  - Default is `20`.

- `--periods D`
  - Holding period in trading days.
  - If omitted, the code uses next-quarter return.
  - If provided, the code uses realized forward return over `D` trading days after quarter end.

Behavior:

- Reads `outputs/models/model_metadata.json`
- Reads `outputs/reports/test_predictions.csv`
- Chooses the signal logic automatically based on the trained target

Writes:

- `outputs/backtests/quarterly_returns.csv`
- `outputs/backtests/portfolio_summary.json`
- `outputs/backtests/portfolio_stats.csv`
- `outputs/backtests/quarterly_returns_plot.svg`
- `outputs/backtests/equity_curves_plot.svg`

### `baseline`

Run the `prelag` baseline through the main CLI.

```bash
pe-pipeline baseline --target-name {target_pe_current,target_pe_scaled_change} [--top-n N] [--start-year YYYY] [--periods D]
```

Optional parameters:

- `--target-name`
  - `target_pe_current`
  - `target_pe_scaled_change`

- `--top-n N`
  - Long basket size and short basket size for the baseline backtest.

- `--start-year YYYY`
  - Rebuild the dataset with a filtered start year before running the baseline.

- `--periods D`
  - Use `D`-trading-day forward return instead of next-quarter return.

Writes into:

- `baseline/prelag/`

### `predict-latest`

Generate the latest-quarter prediction export from the trained main model.

```bash
pe-pipeline predict-latest
```

Writes:

- `data/predictions/latest_predictions.csv`

### `run-all`

Run the main pipeline end to end.

```bash
pe-pipeline run-all --target-name {target_pe_current,target_pe_scaled_change} [--top-n N] [--limit N] [--start-year YYYY] [--periods D]
```

Optional parameters:

- `--target-name`
  - Main training target.

- `--top-n N`
  - Backtest basket size.

- `--limit N`
  - Universe size limit during raw fetch.

- `--start-year YYYY`
  - Dataset start-year filter.

- `--periods D`
  - Backtest holding period in trading days.

Runs:

1. `fetch-data`
2. `build-dataset`
3. `train`
4. `backtest`
5. `predict-latest`

Note:

- `run-all` does not run `tune`.

## Additional Baseline Commands

Two baselines currently live outside the main `pe-pipeline` CLI and are run directly as Python modules.

### Prelag baseline

This is the same baseline behind `pe-pipeline baseline`.

```bash
pe-pipeline baseline --target-name target_pe_current --start-year 2010 --top-n 5 --periods 60
```

Outputs:

- `baseline/prelag/`

### Classification baseline

3-class tree classifier built from the same processed feature set.

```bash
python -m baseline.classification.run_classification --start-year 2010 --periods 60
```

Optional parameters:

- `--start-year YYYY`
- `--periods D`

Outputs:

- `baseline/classification/`

### Lasso baseline

Lasso regression baseline using the same feature set as the tree for direct PE nowcasting.

```bash
python -m baseline.Lasso.run_lasso --start-year 2010 --periods 60 --top-n 5
```

Optional parameters:

- `--start-year YYYY`
- `--periods D`
- `--top-n N`

Outputs:

- `baseline/Lasso/`

## Targets

The main model currently supports two public targets.

### `target_pe_current`

- Direct nowcast of current-quarter PE ratio.

### `target_pe_scaled_change`

- Current-quarter PE change versus previous-quarter PE, scaled by previous-quarter absolute PE magnitude:

```text
(PE_t - PE_{t-1}) / (abs(PE_{t-1}) + epsilon)
```

## Output Locations

### Main model

- Training metrics
  - `outputs/reports/training_metrics.json`

- Hyperparameter search
  - `outputs/reports/best_params.json`
  - `outputs/reports/tuning_results.csv`

- Model metadata
  - `outputs/models/model_metadata.json`

- Main backtest
  - `outputs/backtests/quarterly_returns.csv`
  - `outputs/backtests/portfolio_summary.json`
  - `outputs/backtests/portfolio_stats.csv`

- Latest predictions
  - `data/predictions/latest_predictions.csv`

### Baselines

- Prelag baseline
  - `baseline/prelag/`

- Classification baseline
  - `baseline/classification/`

- Lasso baseline
  - `baseline/Lasso/`

## Tests

You can run repository tests with:

```bash
pytest tests
```

The `tests/` directory is meant to validate the codebase logic, not to hold model outputs.

## Minimal Sanity Checks

```bash
pe-pipeline --help
pe-pipeline build-dataset --help
pe-pipeline tune --help
pe-pipeline backtest --help
pe-pipeline baseline --help
python -m baseline.classification.run_classification --help
python -m baseline.Lasso.run_lasso --help
```
