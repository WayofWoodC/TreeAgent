from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from pe_pipeline.config import get_settings
from pe_pipeline.logging_utils import setup_logging
from pe_pipeline.pipelines.backtest import run_backtest
from pe_pipeline.pipelines.build_dataset import run_build_dataset
from pe_pipeline.pipelines.fetch_data import run_fetch_data
from pe_pipeline.pipelines.predict_latest import run_predict_latest
from pe_pipeline.pipelines.run_all import run_all
from pe_pipeline.pipelines.train import run_train
from pe_pipeline.pipelines.tune import run_tune

try:
    from baseline.prelag.run_baseline import run_baseline
except ModuleNotFoundError:
    baseline_path = Path(__file__).resolve().parents[1] / "baseline" / "prelag" / "run_baseline.py"
    spec = importlib.util.spec_from_file_location("baseline.prelag.run_baseline", baseline_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_baseline = module.run_baseline


def build_parser() -> argparse.ArgumentParser:
    target_choices = ["target_pe_current", "target_pe_scaled_change"]
    parser = argparse.ArgumentParser(description="Quarterly PE ratio prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-data")
    fetch_parser.add_argument("--limit", type=int, default=None)

    dataset_parser = subparsers.add_parser("build-dataset")
    dataset_parser.add_argument("--target-name", choices=target_choices, default="target_pe_scaled_change")
    dataset_parser.add_argument("--start-year", type=int, default=None)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--target-name", choices=target_choices, default="target_pe_scaled_change")

    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--target-name", choices=target_choices, default="target_pe_scaled_change")
    tune_parser.add_argument("--metric", choices=["rmse", "spearman"], default="rmse")
    tune_parser.add_argument("--n-trials", type=int, default=25)

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--top-n", type=int, default=20)
    backtest_parser.add_argument("--periods", type=int, default=None)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--target-name", choices=target_choices, default="target_pe_current")
    baseline_parser.add_argument("--top-n", type=int, default=20)
    baseline_parser.add_argument("--start-year", type=int, default=None)
    baseline_parser.add_argument("--periods", type=int, default=None)

    subparsers.add_parser("predict-latest")

    run_all_parser = subparsers.add_parser("run-all")
    run_all_parser.add_argument("--target-name", choices=target_choices, default="target_pe_scaled_change")
    run_all_parser.add_argument("--top-n", type=int, default=20)
    run_all_parser.add_argument("--limit", type=int, default=None)
    run_all_parser.add_argument("--start-year", type=int, default=None)
    run_all_parser.add_argument("--periods", type=int, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = get_settings()
    logger = setup_logging(settings.outputs_dir / "pipeline.log")

    if args.command == "fetch-data":
        run_fetch_data(settings, logger, limit=args.limit)
    elif args.command == "build-dataset":
        run_build_dataset(settings, logger, target_name=args.target_name, start_year=args.start_year)
    elif args.command == "train":
        run_train(settings, logger, target_name=args.target_name)
    elif args.command == "tune":
        run_tune(settings, logger, target_name=args.target_name, metric=args.metric, n_trials=args.n_trials)
    elif args.command == "backtest":
        run_backtest(settings, logger, top_n=args.top_n, periods=args.periods)
    elif args.command == "baseline":
        run_baseline(
            settings,
            logger,
            target_name=args.target_name,
            top_n=args.top_n,
            start_year=args.start_year,
            periods=args.periods,
        )
    elif args.command == "predict-latest":
        run_predict_latest(settings, logger)
    elif args.command == "run-all":
        run_all(
            settings,
            logger,
            target_name=args.target_name,
            top_n=args.top_n,
            limit=args.limit,
            start_year=args.start_year,
            periods=args.periods,
        )
