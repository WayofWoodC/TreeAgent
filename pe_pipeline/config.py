from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv()


@dataclass(slots=True)
class Settings:
    root_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    intermediate_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    baseline_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    backtests_dir: Path = field(init=False)
    fmp_api_key: str | None = field(default_factory=lambda: os.getenv("FMP_API_KEY"))
    finnhub_api_key: str | None = field(default_factory=lambda: os.getenv("FINNHUB_API_KEY"))
    fred_api_key: str | None = field(default_factory=lambda: os.getenv("FRED_API_KEY"))
    start_date: str = "2000-01-01"
    end_date: str = "2099-12-31"
    epsilon: float = 1.0
    daily_windows: tuple[int, ...] = (5, 20, 60)
    top_n_default: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.intermediate_dir = self.data_dir / "intermediate"
        self.processed_dir = self.data_dir / "processed"
        self.predictions_dir = self.data_dir / "predictions"
        self.outputs_dir = self.root_dir / "outputs"
        self.baseline_dir = self.root_dir / "baseline"
        self.models_dir = self.outputs_dir / "models"
        self.reports_dir = self.outputs_dir / "reports"
        self.backtests_dir = self.outputs_dir / "backtests"


def get_settings() -> Settings:
    return Settings()
