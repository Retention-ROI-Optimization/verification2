from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_ACTIVITY_EVENT_TYPES: tuple[str, ...] = (
    "visit",
    "page_view",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "support_contact",
    "coupon_open",
    "coupon_redeem",
)

COHORT_ACTIVITY_PRESETS: dict[str, tuple[str, ...]] = {
    "all_activity": DEFAULT_ACTIVITY_EVENT_TYPES,
    "core_engagement": (
        "visit",
        "page_view",
        "search",
        "add_to_cart",
        "purchase",
    ),
    "purchase_only": ("purchase",),
    "purchase_or_redeem": ("purchase", "coupon_redeem"),
}

DEFAULT_ACTIVITY_DEFINITION = "core_engagement"
DEFAULT_RETENTION_MODE = "rolling"
VALID_RETENTION_MODES = {"point", "rolling"}


def _month_number_from_string(month_str: pd.Series) -> pd.Series:
    text = month_str.astype(str)
    year = text.str.slice(0, 4).astype(int)
    month = text.str.slice(5, 7).astype(int)
    return year * 12 + month


def _normalize_event_types(event_types: Optional[Iterable[str]]) -> Optional[set[str]]:
    if event_types is None:
        return None
    normalized = {str(x) for x in event_types}
    return normalized or None


def _resolve_activity_event_types(
    activity_definition: str,
    activity_event_types: Optional[Sequence[str]],
) -> tuple[str, Optional[Sequence[str]]]:
    if activity_event_types is not None:
        return activity_definition or "custom", activity_event_types
    definition = activity_definition or DEFAULT_ACTIVITY_DEFINITION
    if definition not in COHORT_ACTIVITY_PRESETS:
        raise ValueError(
            f"Unknown activity_definition='{definition}'. "
            f"Expected one of {sorted(COHORT_ACTIVITY_PRESETS)} or provide activity_event_types explicitly."
        )
    return definition, COHORT_ACTIVITY_PRESETS[definition]


def build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = 7,
    end_date: Optional[str] = None,
    activity_event_types: Optional[Sequence[str]] = None,
    activity_definition: str = DEFAULT_ACTIVITY_DEFINITION,
    retention_mode: str = DEFAULT_RETENTION_MODE,
    min_events_per_period: int = 1,
) -> pd.DataFrame:
    """
    Build a monthly cohort-retention table.

    Why this version is richer than the old implementation:
    - It separates *what counts as retention* via ``activity_definition``.
      Counting coupon opens/support contacts as retention often inflates curves.
    - It separates *how retention is measured* via ``retention_mode``.
      ``point`` = active in that exact month, so reactivation can make the curve rebound.
      ``rolling`` = active in that month or any later observed month, which is monotonic and
      better for comparing cohort decay.
    - Unobserved future periods are kept as NaN instead of 0 to avoid right-censoring bias.
    """
    columns = [
        "cohort_month",
        "period",
        "cohort_size",
        "retained_customers",
        "retention_rate",
        "observed",
        "activity_definition",
        "retention_mode",
        "min_events_per_period",
    ]

    if customers.empty:
        return pd.DataFrame(columns=columns)

    if periods <= 0:
        raise ValueError("periods must be positive.")

    if min_events_per_period <= 0:
        raise ValueError("min_events_per_period must be positive.")

    retention_mode = str(retention_mode or DEFAULT_RETENTION_MODE)
    if retention_mode not in VALID_RETENTION_MODES:
        raise ValueError(
            f"Unknown retention_mode='{retention_mode}'. Expected one of {sorted(VALID_RETENTION_MODES)}."
        )

    activity_definition, activity_event_types = _resolve_activity_event_types(
        activity_definition=activity_definition,
        activity_event_types=activity_event_types,
    )

    base = customers[["customer_id", "acquisition_month"]].copy()
    base["cohort_month"] = base["acquisition_month"].astype(str)
    base["cohort_month_num"] = _month_number_from_string(base["cohort_month"])
    base = base.drop_duplicates(subset=["customer_id"])

    event_type_filter = _normalize_event_types(activity_event_types)

    if events.empty:
        monthly_activity = pd.DataFrame(columns=["customer_id", "event_month_num", "event_count"])
        inferred_end_month_num = int(base["cohort_month_num"].max())
    else:
        activity = events[["customer_id", "timestamp", "event_type"]].copy()
        if event_type_filter is not None:
            activity = activity[activity["event_type"].astype(str).isin(event_type_filter)].copy()

        if activity.empty:
            monthly_activity = pd.DataFrame(columns=["customer_id", "event_month_num", "event_count"])
            inferred_end_month_num = int(base["cohort_month_num"].max())
        else:
            activity["event_time"] = pd.to_datetime(activity["timestamp"], errors="coerce")
            activity = activity.dropna(subset=["event_time"])
            activity["event_month_num"] = (
                activity["event_time"].dt.year * 12 + activity["event_time"].dt.month
            )
            monthly_activity = (
                activity.groupby(["customer_id", "event_month_num"], as_index=False)
                .size()
                .rename(columns={"size": "event_count"})
            )
            monthly_activity = monthly_activity[
                monthly_activity["event_count"] >= int(min_events_per_period)
            ].copy()
            inferred_end_month_num = int(activity["event_month_num"].max())

    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        end_month_num = int(end_ts.year * 12 + end_ts.month)
    else:
        end_month_num = inferred_end_month_num

    merged = base.merge(monthly_activity, on="customer_id", how="left")
    merged["period"] = merged["event_month_num"] - merged["cohort_month_num"]
    merged = merged[(merged["period"] >= 0) & (merged["period"] < periods)].copy()

    cohort_sizes = base.groupby("cohort_month")["customer_id"].nunique()
    observed_max_period = (
        end_month_num - cohort_sizes.index.to_series().pipe(_month_number_from_string)
    ).astype(int)

    if retention_mode == "point":
        retained_counts = merged.groupby(["cohort_month", "period"])["customer_id"].nunique()
        rolling_last_period = None
    else:
        retained_counts = None
        rolling_last_period = merged.groupby(["cohort_month", "customer_id"])["period"].max()

    rows: list[dict] = []
    for cohort_month, cohort_size in cohort_sizes.items():
        max_observed = int(max(observed_max_period.get(cohort_month, 0), 0))
        cohort_size_int = int(cohort_size)

        if retention_mode == "rolling":
            cohort_last_period = rolling_last_period.loc[cohort_month] if cohort_month in rolling_last_period.index.get_level_values(0) else pd.Series(dtype=float)
            cohort_last_period = pd.to_numeric(cohort_last_period, errors="coerce").dropna()
        else:
            cohort_last_period = pd.Series(dtype=float)

        for period in range(periods):
            is_observed = period <= max_observed
            if not is_observed:
                retained_customers = np.nan
                retention_rate = np.nan
            elif period == 0:
                retained_customers = cohort_size_int
                retention_rate = 1.0
            elif retention_mode == "point":
                retained_customers = int(retained_counts.get((cohort_month, period), 0))
                retention_rate = retained_customers / max(cohort_size_int, 1)
            else:
                retained_customers = int((cohort_last_period >= period).sum())
                retention_rate = retained_customers / max(cohort_size_int, 1)

            rows.append(
                {
                    "cohort_month": str(cohort_month),
                    "period": int(period),
                    "cohort_size": cohort_size_int,
                    "retained_customers": retained_customers,
                    "retention_rate": retention_rate,
                    "observed": bool(is_observed),
                    "activity_definition": activity_definition,
                    "retention_mode": retention_mode,
                    "min_events_per_period": int(min_events_per_period),
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values(
        ["activity_definition", "retention_mode", "cohort_month", "period"]
    ).reset_index(drop=True)


def build_all_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = 7,
    end_date: Optional[str] = None,
    activity_presets: Optional[Mapping[str, Sequence[str]]] = None,
    retention_modes: Sequence[str] = ("point", "rolling"),
    min_events_per_period: int = 1,
) -> pd.DataFrame:
    """Build a stacked cohort table for multiple activity definitions and retention modes."""
    presets = dict(activity_presets or COHORT_ACTIVITY_PRESETS)
    tables: list[pd.DataFrame] = []
    for activity_definition, event_types in presets.items():
        for retention_mode in retention_modes:
            tables.append(
                build_cohort_retention(
                    customers=customers,
                    events=events,
                    periods=periods,
                    end_date=end_date,
                    activity_event_types=event_types,
                    activity_definition=activity_definition,
                    retention_mode=retention_mode,
                    min_events_per_period=min_events_per_period,
                )
            )

    if not tables:
        return pd.DataFrame(
            columns=[
                "cohort_month",
                "period",
                "cohort_size",
                "retained_customers",
                "retention_rate",
                "observed",
                "activity_definition",
                "retention_mode",
                "min_events_per_period",
            ]
        )
    return pd.concat(tables, ignore_index=True)


__all__ = [
    "ACTIVITY_DEFINITION_PRESETS",
    "COHORT_ACTIVITY_PRESETS",
    "DEFAULT_ACTIVITY_DEFINITION",
    "DEFAULT_ACTIVITY_EVENT_TYPES",
    "DEFAULT_RETENTION_MODE",
    "VALID_RETENTION_MODES",
    "build_all_cohort_retention",
    "build_cohort_retention",
]
