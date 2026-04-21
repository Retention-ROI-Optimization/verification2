from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.optimization.dose_response import ACTION_INTENSITIES, load_dose_response_policy_model
from src.optimization.timing import apply_survival_timing


INTENSITY_PROFILES = {
    "low": {
        "label": "저강도",
        "cost_multiplier": 0.65,
        "base_effect_multiplier": 0.88,
        "timing_weight": 0.96,
        "response_elasticity": 0.28,
    },
    "mid": {
        "label": "중강도",
        "cost_multiplier": 1.00,
        "base_effect_multiplier": 1.00,
        "timing_weight": 1.00,
        "response_elasticity": 0.22,
    },
    "high": {
        "label": "고강도",
        "cost_multiplier": 1.45,
        "base_effect_multiplier": 0.98,
        "timing_weight": 1.02,
        "response_elasticity": 0.18,
    },
}


SEGMENT_INTENSITY_BIAS = {
    "High Value-Persuadables": {"low": 0.95, "mid": 1.08, "high": 1.18},
    "High Value-Sure Things": {"low": 1.05, "mid": 0.92, "high": 0.78},
    "High Value-Lost Causes": {"low": 0.88, "mid": 0.80, "high": 0.68},
    "Low Value-Persuadables": {"low": 1.10, "mid": 0.98, "high": 0.84},
    "Low Value-Sure Things": {"low": 1.06, "mid": 0.90, "high": 0.74},
    "Low Value-Lost Causes": {"low": 0.82, "mid": 0.72, "high": 0.58},
    "New Customers": {"low": 1.02, "mid": 0.94, "high": 0.86},
}


DEFAULT_LIVE_STRATEGY = {
    "label": "Generic retention offer",
    "base_effect_multiplier": 1.0,
}


def safe_numeric(series: pd.Series | Iterable[float] | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(float(default))


def normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    numeric = safe_numeric(series, default=0.0)
    low = float(numeric.min())
    high = float(numeric.max())
    if high - low < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - low) / (high - low)


def column_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float(default)] * len(df), index=df.index, dtype=float)
    return safe_numeric(df[column], default=default)


def _base_strategy_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strategy_name" not in out.columns:
        out["strategy_name"] = DEFAULT_LIVE_STRATEGY["label"]
    else:
        out["strategy_name"] = out["strategy_name"].fillna(DEFAULT_LIVE_STRATEGY["label"]).astype(str)

    if "strategy_effect_multiplier" in out.columns:
        out["strategy_effect_multiplier"] = safe_numeric(out["strategy_effect_multiplier"], default=1.0).clip(lower=0.1)
    elif "effect_multiplier" in out.columns:
        out["strategy_effect_multiplier"] = safe_numeric(out["effect_multiplier"], default=1.0).clip(lower=0.1)
    else:
        out["strategy_effect_multiplier"] = float(DEFAULT_LIVE_STRATEGY["base_effect_multiplier"])

    if "strategy_cost" not in out.columns:
        if "coupon_cost" in out.columns:
            out["strategy_cost"] = safe_numeric(out["coupon_cost"], default=0.0).clip(lower=0.0)
        else:
            out["strategy_cost"] = 0.0
    else:
        out["strategy_cost"] = safe_numeric(out["strategy_cost"], default=0.0).clip(lower=0.0)
    return out


def _resolve_segment_bias(row: pd.Series, intensity_key: str) -> float:
    segment = str(row.get("customer_segment", ""))
    profile = SEGMENT_INTENSITY_BIAS.get(segment)
    if profile:
        return float(profile.get(intensity_key, 1.0))
    uplift_segment = str(row.get("uplift_segment", ""))
    if uplift_segment == "Persuadables":
        return {"low": 1.02, "mid": 1.08, "high": 1.12}.get(intensity_key, 1.0)
    if uplift_segment == "Sure Things":
        return {"low": 1.04, "mid": 0.92, "high": 0.76}.get(intensity_key, 1.0)
    if uplift_segment in {"Lost Causes", "Sleeping Dogs"}:
        return {"low": 0.84, "mid": 0.74, "high": 0.60}.get(intensity_key, 1.0)
    return 1.0


def _resolve_intensity_profiles(dose_response_model=None) -> dict[str, dict[str, float | str]]:
    profiles = {key: value.copy() for key, value in INTENSITY_PROFILES.items()}
    if dose_response_model is None:
        return profiles
    learned_costs = getattr(dose_response_model, "intensity_cost_multipliers", {}) or {}
    for intensity_key in ACTION_INTENSITIES:
        learned_cost = learned_costs.get(intensity_key)
        if learned_cost is not None and np.isfinite(float(learned_cost)):
            profiles[intensity_key]["cost_multiplier"] = float(np.clip(float(learned_cost), 0.35, 2.10))
    return profiles


def _apply_learned_dose_response(frame: pd.DataFrame, intensity_key: str, dose_response_model) -> pd.DataFrame:
    learned = dose_response_model.predict_effect_frame(frame, intensity_key=intensity_key)
    out = frame.copy()
    out["dose_response_enabled"] = True
    out["dose_response_model_version"] = getattr(dose_response_model, "version", "unknown")
    out["segment_intensity_bias"] = 1.0
    out["dose_response_incremental_effect"] = safe_numeric(learned["dose_response_incremental_effect"], default=0.0).clip(-0.35, 0.75)
    out["dose_response_retention_prob_treated"] = safe_numeric(learned["dose_response_retention_prob_treated"], default=0.0).clip(0.0, 1.0)
    out["dose_response_retention_prob_control"] = safe_numeric(learned["dose_response_retention_prob_control"], default=0.0).clip(0.0, 1.0)

    uplift_anchor = safe_numeric(out.get("uplift_score"), default=0.0).abs().clip(lower=0.02)
    value_basis = column_or_default(out, "predicted_clv_12m", np.nan)
    fallback_clv = column_or_default(out, "clv", np.nan)
    fallback_revenue = column_or_default(out, "base_expected_revenue", 0.0)
    value_basis = value_basis.where(value_basis.notna(), fallback_clv)
    value_basis = value_basis.where(value_basis.notna(), fallback_revenue)
    value_basis = value_basis.clip(lower=0.0)

    out["intensity_effect_multiplier"] = (out["dose_response_incremental_effect"].abs() / uplift_anchor).clip(lower=0.20, upper=2.50)
    out["expected_revenue"] = (
        out["dose_response_incremental_effect"]
        * out["strategy_effect_multiplier"]
        * value_basis
        * out["churn_timing_weight"]
    ).clip(lower=0.0)
    fatigue_guardrail = (
        1.0
        - 0.18 * normalize(column_or_default(out, "discount_pressure_score", 0.0)).clip(0.0, 1.0)
        - 0.05 * normalize(column_or_default(out, "brand_sensitivity", 0.0)).clip(0.0, 1.0)
    ).clip(lower=0.55, upper=1.0)
    out["fatigue_guardrail_multiplier"] = fatigue_guardrail
    out["expected_incremental_profit"] = ((out["expected_revenue"] * fatigue_guardrail) - out["coupon_cost"]).clip(lower=-out["coupon_cost"])
    out["expected_revenue"] = (out["expected_incremental_profit"] + out["coupon_cost"]).clip(lower=0.0)
    out["expected_roi"] = out["expected_incremental_profit"] / out["coupon_cost"].where(out["coupon_cost"] > 0, 1.0)
    return out


def _apply_heuristic_intensity(frame: pd.DataFrame, intensity_key: str, profile: dict[str, float | str]) -> pd.DataFrame:
    out = frame.copy()
    out["dose_response_enabled"] = False
    out["dose_response_model_version"] = "heuristic_fallback"
    out["segment_intensity_bias"] = out.apply(_resolve_segment_bias, axis=1, intensity_key=intensity_key)
    out["intensity_cost_multiplier"] = float(profile["cost_multiplier"])
    out["coupon_cost"] = (out["strategy_cost"] * out["intensity_cost_multiplier"]).round(2)

    response_readiness = (
        0.38 * out["coupon_affinity_norm"]
        + 0.20 * out["price_sensitivity_norm"]
        + 0.22 * out["uplift_score_norm"]
        + 0.20 * out["timing_urgency_score"]
    ).clip(lower=0.0, upper=1.0)
    urgency_pressure = (
        0.55 * out["timing_urgency_score"]
        + 0.25 * out["risk_percentile"]
        + 0.20 * (1.0 - out["window_rank_score"])
    ).clip(lower=0.0, upper=1.0)
    low_intensity_preference = (0.55 * out["coupon_affinity_norm"] + 0.45 * out["uplift_score_norm"]).clip(0.0, 1.0)

    base_effect = float(profile["base_effect_multiplier"])
    timing_weight = float(profile["timing_weight"])
    elasticity = float(profile["response_elasticity"])

    if intensity_key == "low":
        intensity_fit = 0.86 + 0.18 * low_intensity_preference + 0.08 * (1.0 - urgency_pressure)
    elif intensity_key == "mid":
        intensity_fit = 0.94 + 0.12 * response_readiness + 0.06 * urgency_pressure
    else:
        intensity_fit = (
            0.72
            + 0.18 * response_readiness
            + 0.16 * urgency_pressure
            + 0.08 * out["coupon_affinity_norm"]
            + 0.06 * out["price_sensitivity_norm"]
            - 0.14 * (1.0 - response_readiness)
        )

    discount_pressure = normalize(column_or_default(out, "discount_pressure_score", 0.0)).clip(0.0, 1.0)
    fatigue_guardrail = (
        1.0
        - (0.08 if intensity_key == "low" else 0.14 if intensity_key == "mid" else 0.22) * discount_pressure
        - 0.06 * normalize(column_or_default(out, "brand_sensitivity", 0.0)).clip(0.0, 1.0)
    ).clip(lower=0.55, upper=1.0)

    out["intensity_effect_multiplier"] = (
        base_effect
        * timing_weight
        * out["segment_intensity_bias"]
        * intensity_fit
        * (1.0 + elasticity * (response_readiness - 0.5))
        * fatigue_guardrail
    ).clip(lower=0.30, upper=1.25)

    out["fatigue_guardrail_multiplier"] = fatigue_guardrail

    out["expected_incremental_profit"] = (
        out["base_net_incremental_profit"]
        * out["churn_timing_weight"]
        * out["intensity_effect_multiplier"]
        - (out["coupon_cost"] - out["strategy_cost"])
    ).clip(lower=-out["coupon_cost"])
    out["expected_revenue"] = (out["expected_incremental_profit"] + out["coupon_cost"]).clip(lower=0.0)
    out["expected_roi"] = out["expected_incremental_profit"] / out["coupon_cost"].where(out["coupon_cost"] > 0, 1.0)
    out["dose_response_incremental_effect"] = np.nan
    out["dose_response_retention_prob_treated"] = np.nan
    out["dose_response_retention_prob_control"] = np.nan
    return out


def _build_action_rows(base: pd.DataFrame, dose_response_model=None) -> pd.DataFrame:
    if base.empty:
        return base.head(0).copy()

    profiles = _resolve_intensity_profiles(dose_response_model=dose_response_model)
    rows: list[pd.DataFrame] = []
    for intensity_key in ACTION_INTENSITIES:
        profile = profiles[intensity_key]
        frame = base.copy()
        frame["intervention_intensity"] = intensity_key
        frame["intervention_intensity_label"] = str(profile["label"])
        frame["action_id"] = frame["customer_id"].astype(str) + "::" + intensity_key
        frame["intensity_cost_multiplier"] = float(profile["cost_multiplier"])
        frame["coupon_cost"] = (frame["strategy_cost"] * frame["intensity_cost_multiplier"]).round(2)

        if dose_response_model is not None:
            frame = _apply_learned_dose_response(frame, intensity_key=intensity_key, dose_response_model=dose_response_model)
        else:
            frame = _apply_heuristic_intensity(frame, intensity_key=intensity_key, profile=profile)

        frame["recommended_action"] = (
            frame["strategy_name"].astype(str)
            + " · "
            + frame["intervention_intensity_label"].astype(str)
            + " · "
            + frame["recommended_intervention_window"].astype(str)
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_intensity_action_candidates(
    df: pd.DataFrame,
    survival_predictions: Optional[pd.DataFrame] = None,
    *,
    customer_id_col: str = "customer_id",
    dose_response_model=None,
    use_learned_dose_response: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.head(0).copy()

    base = _base_strategy_profile(df)
    base = apply_survival_timing(base, survival_predictions=survival_predictions, customer_id_col=customer_id_col)

    base["customer_id"] = pd.to_numeric(base[customer_id_col], errors="coerce")
    base = base.dropna(subset=["customer_id"]).copy()
    if base.empty:
        return base
    base["customer_id"] = base["customer_id"].astype(int)

    base["coupon_affinity_norm"] = normalize(column_or_default(base, "coupon_affinity", 0.0)).clip(0.0, 1.0)
    base["price_sensitivity_norm"] = normalize(column_or_default(base, "price_sensitivity", 0.0)).clip(0.0, 1.0)
    base["uplift_score_norm"] = normalize(column_or_default(base, "uplift_score", 0.0)).clip(0.0, 1.0)
    base["window_rank_score"] = normalize(column_or_default(base, "intervention_window_days", 90.0)).clip(0.0, 1.0)

    baseline_profit = column_or_default(base, "expected_incremental_profit", 0.0)
    baseline_cost = column_or_default(base, "strategy_cost", 0.0)
    baseline_revenue = column_or_default(base, "base_expected_revenue", np.nan)
    baseline_revenue = baseline_revenue.where(baseline_revenue.notna(), baseline_profit + baseline_cost)
    base["base_expected_revenue"] = baseline_revenue.clip(lower=0.0)
    base["base_net_incremental_profit"] = baseline_profit.clip(lower=0.0)

    resolved_dose_model = dose_response_model
    if resolved_dose_model is None and use_learned_dose_response:
        resolved_dose_model = load_dose_response_policy_model()

    action_candidates = _build_action_rows(base, dose_response_model=resolved_dose_model)
    action_candidates["roi_rank_score"] = normalize(action_candidates["expected_roi"])
    action_candidates["profit_rank_score"] = normalize(action_candidates["expected_incremental_profit"])
    predicted_clv = column_or_default(action_candidates, "predicted_clv_12m", np.nan)
    fallback_clv = column_or_default(action_candidates, "clv", 0.0)
    action_candidates["value_rank_score"] = normalize(predicted_clv.where(predicted_clv.notna(), fallback_clv))
    discount_penalty = normalize(column_or_default(action_candidates, "discount_pressure_score", 0.0)).clip(0.0, 1.0)
    action_candidates["priority_score"] = (
        0.26 * action_candidates["roi_rank_score"]
        + 0.24 * action_candidates["profit_rank_score"]
        + 0.14 * safe_numeric(action_candidates.get("churn_probability"), default=0.0).clip(0.0, 1.0)
        + 0.10 * safe_numeric(action_candidates.get("uplift_score"), default=0.0).clip(lower=0.0)
        + 0.12 * action_candidates["timing_urgency_score"]
        + 0.08 * (1.0 - action_candidates["window_rank_score"])
        + 0.06 * action_candidates["value_rank_score"]
        - 0.08 * discount_penalty
    )
    return action_candidates
