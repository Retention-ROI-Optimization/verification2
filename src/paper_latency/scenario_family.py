from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCENARIO_FAMILIES = {
    'complaint-heavy',
    'promotion-heavy',
    'dormancy-heavy',
    'seasonal-shift',
}


@dataclass(frozen=True)
class ScenarioResult:
    features: pd.DataFrame
    metadata: dict[str, float | str | int]


def _safe_col(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float(default)] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce').fillna(float(default))


def _month_factor(decision_date: pd.Timestamp) -> float:
    month = int(pd.Timestamp(decision_date).month)
    # Stronger shifts near holiday / seasonal campaign windows.
    return {1: 0.96, 2: 0.98, 3: 1.00, 4: 1.01, 5: 1.02, 6: 1.04, 7: 1.06, 8: 1.07, 9: 1.05, 10: 1.03, 11: 1.08, 12: 1.12}.get(month, 1.0)


def apply_scenario_family(features: pd.DataFrame, family: str, decision_date: str | pd.Timestamp) -> ScenarioResult:
    family = str(family).strip().lower()
    if family not in SCENARIO_FAMILIES:
        raise ValueError(f'Unsupported scenario family: {family}')

    decision_ts = pd.Timestamp(decision_date)
    out = features.copy()
    meta: dict[str, float | str | int] = {'scenario_family': family, 'decision_date': str(decision_ts.date())}

    if family == 'complaint-heavy':
        support = _safe_col(out, 'support_contact_30d')
        support_rate = _safe_col(out, 'support_contact_rate_30d')
        purchase_drop = _safe_col(out, 'purchase_change_rate_14d')
        out['support_contact_30d'] = np.round(support * 1.45 + 1.0)
        out['support_contact_rate_30d'] = (support_rate * 1.35 + 0.015).clip(lower=0.0)
        out['purchase_change_rate_14d'] = (purchase_drop - 0.10).clip(lower=-1.0, upper=2.5)
        out['pageviews_change_rate'] = (_safe_col(out, 'pageviews_change_rate') - 0.05).clip(lower=-1.0, upper=2.5)
        meta['support_multiplier'] = 1.45

    elif family == 'promotion-heavy':
        exposure = _safe_col(out, 'exposure_count_30d')
        open_rate = _safe_col(out, 'coupon_open_rate_30d')
        coupon_cost = _safe_col(out, 'coupon_cost_30d')
        out['exposure_count_30d'] = np.round(exposure * 1.65 + 1.0)
        out['coupon_open_30d'] = np.round(_safe_col(out, 'coupon_open_30d') * 1.55 + 1.0)
        out['coupon_open_rate_30d'] = (open_rate * 1.18 + 0.02).clip(lower=0.0, upper=1.0)
        out['coupon_response_change_rate'] = (_safe_col(out, 'coupon_response_change_rate') + 0.06).clip(lower=-1.0, upper=2.5)
        out['coupon_cost_30d'] = (coupon_cost * 1.25 + 500.0).clip(lower=0.0)
        meta['promotion_pressure'] = 1.65

    elif family == 'dormancy-heavy':
        inactivity = _safe_col(out, 'inactivity_days')
        visits = _safe_col(out, 'visits_14d')
        purchases = _safe_col(out, 'purchases_14d')
        out['inactivity_days'] = np.round(inactivity + 4.0)
        out['days_since_last_event'] = np.round(_safe_col(out, 'days_since_last_event') + 3.0)
        out['visits_14d'] = np.round(visits * 0.72)
        out['purchases_14d'] = np.round(purchases * 0.70)
        out['visit_change_rate_14d'] = (_safe_col(out, 'visit_change_rate_14d') - 0.18).clip(lower=-1.0, upper=2.5)
        out['purchase_change_rate_14d'] = (_safe_col(out, 'purchase_change_rate_14d') - 0.16).clip(lower=-1.0, upper=2.5)
        out['active_days_30d'] = np.round(_safe_col(out, 'active_days_30d') * 0.82)
        meta['dormancy_shift_days'] = 4

    elif family == 'seasonal-shift':
        factor = _month_factor(decision_ts)
        out['visits_14d'] = np.round(_safe_col(out, 'visits_14d') * factor)
        out['purchases_14d'] = np.round(_safe_col(out, 'purchases_14d') * (factor * 0.96))
        out['weekend_activity_ratio'] = (_safe_col(out, 'weekend_activity_ratio') * min(factor, 1.12)).clip(lower=0.0, upper=1.0)
        out['weekend_purchase_ratio'] = (_safe_col(out, 'weekend_purchase_ratio') * min(factor, 1.10)).clip(lower=0.0, upper=1.0)
        out['session_duration_change_rate'] = (_safe_col(out, 'session_duration_change_rate') + (factor - 1.0) * 0.25).clip(lower=-1.0, upper=2.5)
        out['search_purchase_conv_change_rate'] = (_safe_col(out, 'search_purchase_conv_change_rate') + (factor - 1.0) * 0.20).clip(lower=-1.0, upper=2.5)
        meta['seasonality_factor'] = round(float(factor), 4)

    # Keep a scenario-specific urgency feature for downstream engine logic.
    out['scenario_family'] = family
    out['scenario_month'] = int(decision_ts.month)
    out['scenario_weekofyear'] = int(decision_ts.isocalendar().week)
    return ScenarioResult(features=out, metadata=meta)
