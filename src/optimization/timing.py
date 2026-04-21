from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_SURVIVAL_HORIZON_DAYS = 90
SURVIVAL_FILENAME = "survival_predictions.csv"


SURVIVAL_REQUIRED_COLUMNS = [
    'customer_id',
    'predicted_hazard_ratio',
    'predicted_median_time_to_churn_days',
    'risk_percentile',
    'risk_group',
]


TIMING_EXPORT_COLUMNS = [
    'predicted_hazard_ratio',
    'predicted_median_time_to_churn_days',
    'risk_percentile',
    'risk_group',
    'short_term_survival_probability',
    'short_term_churn_probability',
    'mid_term_survival_probability',
    'mid_term_churn_probability',
    'timing_urgency_score',
    'churn_timing_weight',
    'intervention_window_days',
    'recommended_intervention_window',
    'timing_priority_bucket',
]


def _safe_numeric(series: pd.Series | Iterable[float] | None, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors='coerce').fillna(float(default))




def _column_or_default(df: pd.DataFrame, column: str | None, default: float) -> pd.Series:
    if column and column in df.columns:
        return _safe_numeric(df[column], default)
    return pd.Series([float(default)] * len(df), index=df.index, dtype=float)

def _normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    numeric = _safe_numeric(series)
    low = float(numeric.min())
    high = float(numeric.max())
    if high - low < 1e-12:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index, dtype=float)
    return (numeric - low) / (high - low)


def _extract_survival_days(columns: Iterable[str]) -> list[int]:
    days: list[int] = []
    for column in columns:
        match = re.fullmatch(r'survival_prob_(\d+)d', str(column))
        if match:
            days.append(int(match.group(1)))
    return sorted(set(days))


def _nearest_survival_column(columns: Iterable[str], target_day: int) -> str | None:
    candidates = _extract_survival_days(columns)
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: (abs(value - int(target_day)), value))
    return f'survival_prob_{int(nearest)}d'


def load_survival_predictions(result_dir: str | Path | None) -> pd.DataFrame:
    if result_dir is None:
        return pd.DataFrame()
    path = Path(result_dir) / SURVIVAL_FILENAME
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if 'customer_id' not in df.columns:
        return pd.DataFrame()
    return df


def _default_timing_frame(df: pd.DataFrame, *, horizon_days: int) -> pd.DataFrame:
    out = df.copy()
    out['predicted_hazard_ratio'] = 1.0
    out['predicted_median_time_to_churn_days'] = float(horizon_days)
    out['risk_percentile'] = 0.0
    out['risk_group'] = 'Unknown'
    out['short_term_survival_probability'] = 1.0
    out['short_term_churn_probability'] = 0.0
    out['mid_term_survival_probability'] = 1.0
    out['mid_term_churn_probability'] = 0.0
    out['timing_urgency_score'] = 0.0
    out['churn_timing_weight'] = 1.0
    out['intervention_window_days'] = int(horizon_days)
    out['recommended_intervention_window'] = f'Monitor (>{min(int(horizon_days), 60)}d)'
    out['timing_priority_bucket'] = 'monitor'
    return out


def apply_survival_timing(
    df: pd.DataFrame,
    survival_predictions: Optional[pd.DataFrame] = None,
    *,
    customer_id_col: str = 'customer_id',
    default_horizon_days: int = DEFAULT_SURVIVAL_HORIZON_DAYS,
) -> pd.DataFrame:
    if df.empty or customer_id_col not in df.columns:
        return _default_timing_frame(df, horizon_days=int(default_horizon_days))

    out = df.copy()
    out[customer_id_col] = pd.to_numeric(out[customer_id_col], errors='coerce')
    out = out.dropna(subset=[customer_id_col]).copy()
    out[customer_id_col] = out[customer_id_col].astype(int)

    if survival_predictions is None or survival_predictions.empty:
        return _default_timing_frame(out, horizon_days=int(default_horizon_days))

    surv = survival_predictions.copy()
    if customer_id_col not in surv.columns:
        return _default_timing_frame(out, horizon_days=int(default_horizon_days))

    surv[customer_id_col] = pd.to_numeric(surv[customer_id_col], errors='coerce')
    surv = surv.dropna(subset=[customer_id_col]).copy()
    if surv.empty:
        return _default_timing_frame(out, horizon_days=int(default_horizon_days))
    surv[customer_id_col] = surv[customer_id_col].astype(int)

    survival_prob_cols = [column for column in surv.columns if re.fullmatch(r'survival_prob_\d+d', str(column))]
    keep_columns = [customer_id_col] + [column for column in SURVIVAL_REQUIRED_COLUMNS if column in surv.columns and column != customer_id_col] + survival_prob_cols
    surv = surv.loc[:, list(dict.fromkeys(keep_columns))].copy()

    out = out.merge(surv, on=customer_id_col, how='left')
    available_days = _extract_survival_days(out.columns)
    reference_horizon = int(max(available_days) if available_days else int(default_horizon_days))
    short_col = _nearest_survival_column(out.columns, min(30, reference_horizon))
    mid_col = _nearest_survival_column(out.columns, min(60, reference_horizon)) or short_col

    out['predicted_hazard_ratio'] = _column_or_default(out, 'predicted_hazard_ratio', 1.0).clip(lower=0.0)
    out['predicted_median_time_to_churn_days'] = _column_or_default(
        out,
        'predicted_median_time_to_churn_days',
        float(reference_horizon),
    ).clip(lower=1.0, upper=float(reference_horizon))

    if 'risk_percentile' in out.columns:
        risk_percentile = _safe_numeric(out['risk_percentile'], np.nan)
        risk_percentile = risk_percentile.where(risk_percentile.notna(), _normalize(np.log1p(out['predicted_hazard_ratio'])))
    else:
        risk_percentile = _normalize(np.log1p(out['predicted_hazard_ratio']))
    out['risk_percentile'] = risk_percentile.clip(lower=0.0, upper=1.0)
    if 'risk_group' in out.columns:
        out['risk_group'] = out['risk_group'].fillna('Unknown').astype(str)
    else:
        out['risk_group'] = 'Unknown'

    short_survival = _column_or_default(out, short_col, 1.0).clip(lower=0.0, upper=1.0)
    mid_survival = _column_or_default(out, mid_col, 1.0).clip(lower=0.0, upper=1.0)
    inverse_time_score = 1.0 - (out['predicted_median_time_to_churn_days'] / max(float(reference_horizon), 1.0)).clip(lower=0.0, upper=1.0)

    timing_urgency_score = (
        0.45 * out['risk_percentile']
        + 0.35 * (1.0 - short_survival)
        + 0.20 * inverse_time_score
    ).clip(lower=0.0, upper=1.0)

    out['short_term_survival_probability'] = short_survival
    out['short_term_churn_probability'] = (1.0 - short_survival).clip(lower=0.0, upper=1.0)
    out['mid_term_survival_probability'] = mid_survival
    out['mid_term_churn_probability'] = (1.0 - mid_survival).clip(lower=0.0, upper=1.0)
    out['timing_urgency_score'] = timing_urgency_score.round(6)
    out['churn_timing_weight'] = (0.85 + 0.60 * timing_urgency_score).round(6)
    out['intervention_window_days'] = out['predicted_median_time_to_churn_days'].round().astype(int).clip(lower=1, upper=reference_horizon)

    window_days = out['intervention_window_days']
    out['recommended_intervention_window'] = np.select(
        [
            window_days <= 14,
            (window_days > 14) & (window_days <= 30),
            (window_days > 30) & (window_days <= 60),
        ],
        [
            'Immediate (<=14d)',
            'Near-term (15-30d)',
            'Planned (31-60d)',
        ],
        default=f'Monitor (>{min(reference_horizon, 60)}d)',
    )
    out['timing_priority_bucket'] = np.select(
        [
            window_days <= 14,
            (window_days > 14) & (window_days <= 30),
            (window_days > 30) & (window_days <= 60),
        ],
        ['immediate', 'near_term', 'planned'],
        default='monitor',
    )
    return out
