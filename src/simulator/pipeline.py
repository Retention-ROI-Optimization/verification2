from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, SimulationConfig
from .customer_generator import generate_customers
from .cohort_analysis import build_all_cohort_retention
from .event_engine import simulate_events
from .exporter import export_tables
from .personas import DEFAULT_PERSONAS
from .treatment import assign_treatment


def _safe_div(numer, denom):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    return numer / np.maximum(denom, 1.0)


def _calibrate_churn_probability(
    raw_score: pd.Series | np.ndarray,
    inactivity_days: pd.Series | np.ndarray,
    churn_threshold: int,
) -> np.ndarray:
    """
    Keep the synthetic churn score monotonic with the latent risk signal,
    while calibrating the 0.50 threshold to roughly match the simulator's
    realized churn share.

    Without this step, the latent score can make more than half of customers
    appear to be "at risk" even when the realized inactivity-based churn rate
    is only around 15~25%.
    """
    raw = np.asarray(raw_score, dtype=float)
    inactivity = np.asarray(inactivity_days, dtype=float)

    if raw.size == 0:
        return np.array([], dtype=float)

    actual_churn_share = float(np.clip(np.mean(inactivity >= float(churn_threshold)), 0.15, 0.25))
    rank_pct = pd.Series(raw).rank(pct=True, method="average").to_numpy(dtype=float)

    center = 1.0 - actual_churn_share
    width = 0.10
    calibrated = 1.0 / (1.0 + np.exp(-(rank_pct - center) / width))
    return np.clip(calibrated, 0.01, 0.99)


def _build_customer_summary(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
    exposures: pd.DataFrame,
    final_state: pd.DataFrame,
    config: SimulationConfig,
) -> pd.DataFrame:
    end_ts = pd.Timestamp(config.end_date)

    visit_events = events[events["event_type"] == "visit"].copy() if not events.empty else pd.DataFrame(columns=["customer_id", "timestamp"])
    if not visit_events.empty:
        visit_events["date"] = pd.to_datetime(visit_events["timestamp"]).dt.normalize()

    purchase_events = orders.copy()
    if not purchase_events.empty:
        purchase_events["date"] = pd.to_datetime(purchase_events["order_time"]).dt.normalize()

    def _count_in_window(df: pd.DataFrame, date_col: str, start_days_ago: int, end_days_ago: int) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        start = end_ts - pd.Timedelta(days=start_days_ago)
        end = end_ts - pd.Timedelta(days=end_days_ago)
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        return df.loc[mask].groupby("customer_id").size()

    visits_last_7 = _count_in_window(visit_events, "date", 6, 0).rename("visits_last_7")
    visits_prev_7 = _count_in_window(visit_events, "date", 13, 7).rename("visits_prev_7")
    purchase_last_30 = _count_in_window(purchase_events, "date", 29, 0).rename("purchase_last_30")
    purchase_prev_30 = _count_in_window(purchase_events, "date", 59, 30).rename("purchase_prev_30")

    coupon_redeem_count = (
        events.loc[events["event_type"] == "coupon_redeem"].groupby("customer_id").size().rename("coupon_redeem_count")
        if not events.empty else pd.Series(dtype=float)
    )
    exposure_count = (
        exposures.groupby("customer_id").size().rename("coupon_exposure_count")
        if not exposures.empty else pd.Series(dtype=float)
    )

    summary = customers.merge(assignments, on="customer_id", how="left").merge(final_state, on="customer_id", how="left")

    for series in [visits_last_7, visits_prev_7, purchase_last_30, purchase_prev_30, coupon_redeem_count, exposure_count]:
        summary = summary.merge(series, on="customer_id", how="left")

    fill_zero_cols = [
        "frequency",
        "monetary",
        "coupon_exposures",
        "coupon_opens",
        "coupon_redeemed",
        "visits_last_7",
        "visits_prev_7",
        "purchase_last_30",
        "purchase_prev_30",
        "coupon_redeem_count",
        "coupon_exposure_count",
    ]
    for col in fill_zero_cols:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)

    summary["visit_change_rate"] = _safe_div(summary["visits_last_7"] - summary["visits_prev_7"], summary["visits_prev_7"])
    summary["purchase_change_rate"] = _safe_div(summary["purchase_last_30"] - summary["purchase_prev_30"], summary["purchase_prev_30"])

    observed_coupon_response = (summary["coupon_redeem_count"] + 1.0) / (summary["coupon_exposure_count"] + 4.0)
    uplift_segment_adjust = np.select(
        [
            summary["uplift_segment_true"] == "persuadable",
            summary["uplift_segment_true"] == "sure_thing",
            summary["uplift_segment_true"] == "lost_cause",
            summary["uplift_segment_true"] == "sleeping_dog",
        ],
        [0.05, -0.01, -0.04, -0.09],
        default=0.0,
    )
    latent_uplift = (
        summary["treatment_lift_base"]
        + 0.05 * summary["coupon_affinity"]
        - 0.04 * summary["price_sensitivity"]
        + uplift_segment_adjust
    )
    summary["uplift_score"] = np.clip(
        0.55 * latent_uplift
        + 0.20 * observed_coupon_response
        + 0.08 * (summary["coupon_exposure_count"] > 0).astype(float),
        -0.15,
        0.42,
    )

    monetary_scaled = np.clip(summary["monetary"] / np.maximum(summary["monetary"].quantile(0.95), 1), 0, 1)
    frequency_scaled = np.clip(summary["frequency"] / np.maximum(summary["frequency"].quantile(0.95), 1), 0, 1)
    recency_scaled = np.clip(summary["recency_days"] / max(config.churn_inactivity_days * 1.5, 1), 0, 1)

    customer_age_days = (end_ts - pd.to_datetime(summary["signup_date"]).dt.normalize()).dt.days.clip(lower=0)
    new_signup_uncertainty = np.where(summary["persona"] == "new_signup", np.clip((120 - customer_age_days) / 120, 0, 1), 0.0)

    persona_boost = (
        np.where(summary["persona"] == "vip_loyal", -0.12, 0.0)
        + np.where(summary["persona"] == "regular_loyal", -0.06, 0.0)
        + np.where(summary["persona"] == "price_sensitive", 0.04, 0.0)
        + np.where(summary["persona"] == "explorer", 0.06, 0.0)
        + np.where(summary["persona"] == "churn_progressing", 0.18, 0.0)
        + np.where(summary["persona"] == "new_signup", 0.05, 0.0)
    )

    base_churn = (
        0.34 * recency_scaled
        + 0.20 * (1 - frequency_scaled)
        + 0.16 * (1 - monetary_scaled)
        + 0.13 * (summary["visit_change_rate"] < 0).astype(float)
        + 0.13 * (summary["purchase_change_rate"] < 0).astype(float)
        + 0.04 * np.clip(summary["inactivity_days"] / max(config.churn_inactivity_days, 1), 0, 1)
        + 0.05 * new_signup_uncertainty
    )
    raw_churn_score = np.clip(base_churn + persona_boost, 0.01, 0.99)
    summary["churn_probability"] = _calibrate_churn_probability(
        raw_score=raw_churn_score,
        inactivity_days=summary["inactivity_days"],
        churn_threshold=config.churn_inactivity_days,
    )

    avg_order_value = _safe_div(summary["monetary"], summary["frequency"])
    retention_factor = np.clip(1.15 - summary["churn_probability"], 0.20, 1.15)
    summary["clv"] = (
        summary["monetary"] * (1.30 + 1.25 * retention_factor)
        + summary["frequency"] * np.maximum(avg_order_value, 20000) * 0.55
    ).clip(lower=15000)

    fatigue = pd.to_numeric(summary.get("coupon_fatigue_score", 0.0), errors="coerce").fillna(0.0)
    fatigue_sensitivity = pd.to_numeric(summary.get("discount_fatigue_sensitivity", 0.0), errors="coerce").fillna(0.0)
    brand_sensitivity = pd.to_numeric(summary.get("brand_sensitivity", 0.0), errors="coerce").fillna(0.0)
    dependency = pd.to_numeric(summary.get("discount_dependency_score", 0.0), errors="coerce").fillna(0.0)
    summary["discount_pressure_score"] = (fatigue * (0.55 + 0.45 * fatigue_sensitivity) + 0.60 * dependency).clip(lower=0.0)
    summary["discount_effect_penalty"] = np.clip(1.0 - 0.08 * summary["discount_pressure_score"] - 0.05 * brand_sensitivity, 0.55, 1.0)

    summary["expected_incremental_profit"] = np.maximum(summary["clv"] * summary["uplift_score"] * summary["discount_effect_penalty"], -50000)
    summary["expected_roi"] = _safe_div(summary["expected_incremental_profit"] - summary["coupon_cost"], summary["coupon_cost"])

    summary["uplift_segment"] = np.select(
        [
            summary["uplift_score"] >= 0.12,
            (summary["uplift_score"] >= 0.02) & (summary["uplift_score"] < 0.12),
            (summary["uplift_score"] >= -0.03) & (summary["uplift_score"] < 0.02),
        ],
        ["Persuadables", "Sure Things", "Lost Causes"],
        default="Sleeping Dogs",
    )

    columns = [
        "customer_id",
        "persona",
        "uplift_segment_true",
        "acquisition_month",
        "recency_days",
        "frequency",
        "monetary",
        "visits_last_7",
        "visits_prev_7",
        "visit_change_rate",
        "purchase_last_30",
        "purchase_prev_30",
        "purchase_change_rate",
        "churn_probability",
        "uplift_score",
        "clv",
        "coupon_cost",
        "expected_incremental_profit",
        "expected_roi",
        "uplift_segment",
        "signup_date",
        "region",
        "device_type",
        "acquisition_channel",
        "treatment_group",
        "treatment_flag",
        "coupon_exposure_count",
        "coupon_redeem_count",
        "inactivity_days",
        "coupon_fatigue_score",
        "discount_dependency_score",
        "discount_pressure_score",
        "discount_effect_penalty",
        "discount_fatigue_sensitivity",
        "offer_dependency_risk",
        "brand_sensitivity",
    ]

    ordered = [c for c in columns if c in summary.columns]
    others = [c for c in summary.columns if c not in ordered]
    summary = summary[ordered + others].sort_values("customer_id").reset_index(drop=True)
    return summary


def _build_cohort_retention(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    periods: int = 13,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    return build_all_cohort_retention(
        customers=customers,
        events=events,
        periods=periods,
        end_date=end_date,
    )


def run_simulation(
    config: Optional[SimulationConfig] = None,
    export: bool = False,
    output_dir: Optional[str] = None,
    file_format: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full simulator pipeline.

    Returned tables:
    - customers
    - treatment_assignments
    - campaign_exposures
    - events
    - orders
    - state_snapshots
    - customer_summary
    - cohort_retention
    """
    config = config or DEFAULT_CONFIG
    rng = np.random.default_rng(config.random_seed)

    customers = generate_customers(config=config, personas=DEFAULT_PERSONAS, rng=rng)
    assignments = assign_treatment(customers=customers, config=config, rng=rng)
    events, orders, exposures, state_snapshots, final_state = simulate_events(
        customers=customers,
        assignments=assignments,
        config=config,
        rng=rng,
    )

    customer_summary = _build_customer_summary(
        customers=customers,
        assignments=assignments,
        events=events,
        orders=orders,
        exposures=exposures,
        final_state=final_state,
        config=config,
    )
    cohort_retention = _build_cohort_retention(
        customers=customers,
        events=events,
        periods=7,
        end_date=config.end_date,
    )

    tables: Dict[str, pd.DataFrame] = {
        "customers": customers,
        "treatment_assignments": assignments,
        "campaign_exposures": exposures,
        "events": events,
        "orders": orders,
        "state_snapshots": state_snapshots,
        "customer_summary": customer_summary,
        "cohort_retention": cohort_retention,
    }

    if export:
        export_tables(
            tables=tables,
            output_dir=output_dir or config.default_export_dir,
            file_format=file_format or config.default_file_format,
        )

    return tables


def run_simulation_for_dashboard(
    config: Optional[SimulationConfig] = None,
    export: bool = False,
    output_dir: Optional[str] = None,
    file_format: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper for the current Streamlit UI.

    Returns:
    - customer_summary
    - cohort_retention
    """
    tables = run_simulation(
        config=config,
        export=export,
        output_dir=output_dir,
        file_format=file_format,
    )
    return tables["customer_summary"], tables["cohort_retention"]
