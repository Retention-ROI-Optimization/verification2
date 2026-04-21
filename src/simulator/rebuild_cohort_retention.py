from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cohort_analysis import build_all_cohort_retention
from .config import DEFAULT_CONFIG


def rebuild_cohort_retention(
    data_dir: str = "data/raw",
    periods: int = 13,
    end_date: str | None = None,
) -> Path:
    data_path = Path(data_dir)
    customers_path = data_path / "customers.csv"
    events_path = data_path / "events.csv"
    output_path = data_path / "cohort_retention.csv"

    customers = pd.read_csv(customers_path)
    events = pd.read_csv(events_path)

    cohort_retention = build_all_cohort_retention(
        customers=customers,
        events=events,
        periods=periods,
        end_date=end_date or DEFAULT_CONFIG.end_date,
    )
    cohort_retention.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = rebuild_cohort_retention()
    print(f"cohort_retention rebuilt at: {path}")
