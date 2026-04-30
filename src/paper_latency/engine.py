from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.policy import build_intensity_action_candidates, normalize


STRATEGY_BY_SEGMENT = {
    'High Value-Persuadables': {'strategy_name': 'VIP concierge + personalized offer', 'strategy_cost': 30000.0, 'strategy_effect_multiplier': 1.15},
    'High Value-Sure Things': {'strategy_name': 'Loyalty touchpoint', 'strategy_cost': 8000.0, 'strategy_effect_multiplier': 0.15},
    'High Value-Lost Causes': {'strategy_name': 'Deep-dive outreach', 'strategy_cost': 12000.0, 'strategy_effect_multiplier': 0.10},
    'Low Value-Persuadables': {'strategy_name': 'Coupon campaign', 'strategy_cost': 7000.0, 'strategy_effect_multiplier': 0.85},
    'Low Value-Lost Causes': {'strategy_name': 'No Action', 'strategy_cost': 0.0, 'strategy_effect_multiplier': 0.0},
    'Low Value-Sure Things': {'strategy_name': 'Light reminder', 'strategy_cost': 3000.0, 'strategy_effect_multiplier': 0.05},
    'New Customers': {'strategy_name': 'Onboarding sequence', 'strategy_cost': 5000.0, 'strategy_effect_multiplier': 0.20},
}


@dataclass
class PolicySelection:
    budget: int
    candidates: pd.DataFrame
    selected: pd.DataFrame
    summary: dict[str, Any]



def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float(default)] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce').fillna(float(default))



def _clip01(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    values = pd.Series(series)
    return pd.to_numeric(values, errors='coerce').fillna(0.0).clip(lower=0.0, upper=1.0)



def _logistic(x: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))



def compute_uplift_score(frame: pd.DataFrame, *, scenario_family: str) -> pd.Series:
    coupon_affinity = _clip01(_series(frame, 'coupon_affinity'))
    price_sensitivity = _clip01(_series(frame, 'price_sensitivity'))
    treatment_lift = _series(frame, 'treatment_lift_base')
    coupon_open_rate = _clip01(_series(frame, 'coupon_open_rate_30d'))
    coupon_response_change = _series(frame, 'coupon_response_change_rate').clip(lower=-1.0, upper=2.0)
    cart_conversion = _series(frame, 'cart_to_purchase_rate_30d').clip(lower=0.0)
    fatigue = normalize(_series(frame, 'discount_pressure_score', default=0.0)).clip(0.0, 1.0)
    brand_sensitivity = normalize(_series(frame, 'brand_sensitivity', default=0.0)).clip(0.0, 1.0)
    support_pressure = normalize(_series(frame, 'support_contact_30d', default=0.0)).clip(0.0, 1.0)

    uplift = (
        0.22 * coupon_affinity
        + 0.18 * _clip01(coupon_open_rate)
        + 0.16 * np.tanh(coupon_response_change)
        + 0.15 * np.tanh(treatment_lift * 2.5)
        + 0.10 * np.tanh(cart_conversion)
        - 0.18 * price_sensitivity
        - 0.14 * fatigue
        - 0.08 * brand_sensitivity
    )

    if scenario_family == 'complaint-heavy':
        uplift = uplift - 0.05 * support_pressure
    elif scenario_family == 'promotion-heavy':
        uplift = uplift + 0.04 * coupon_affinity - 0.03 * fatigue
    elif scenario_family == 'dormancy-heavy':
        uplift = uplift - 0.03 * normalize(_series(frame, 'inactivity_days')).clip(0.0, 1.0)
    elif scenario_family == 'seasonal-shift':
        uplift = uplift + 0.02 * _clip01(_series(frame, 'weekend_activity_ratio'))

    return pd.Series(np.clip(uplift, -0.15, 0.40), index=frame.index, dtype=float)



def compute_predicted_clv(frame: pd.DataFrame) -> pd.Series:
    monetary_90d = _series(frame, 'monetary_90d')
    monetary_30d = _series(frame, 'monetary_30d')
    frequency_90d = _series(frame, 'frequency_90d')
    avg_order_value = _series(frame, 'avg_order_value_90d')
    customer_age_days = _series(frame, 'customer_age_days')
    active_days = _series(frame, 'active_days_30d')
    recency_days = _series(frame, 'recency_days')

    retention_potential = (1.25 + 0.40 * normalize(active_days) - 0.20 * normalize(recency_days)).clip(lower=0.50, upper=1.60)
    clv = (
        monetary_90d * (2.4 + 0.6 * retention_potential)
        + monetary_30d * 1.1
        + frequency_90d * np.maximum(avg_order_value, 15000.0) * 0.55
        + customer_age_days * 12.0
    )
    return pd.Series(np.clip(clv, 15000.0, None), index=frame.index, dtype=float)



def compute_survival_predictions(frame: pd.DataFrame, churn_probability: pd.Series, *, horizon_days: int = 90) -> pd.DataFrame:
    churn_prob = _clip01(churn_probability)
    inactivity = _series(frame, 'inactivity_days').clip(lower=0.0)
    purchase_anomaly = _series(frame, 'purchase_cycle_anomaly').clip(lower=0.0)
    visit_drop = _series(frame, 'visit_change_rate_14d').clip(lower=-1.0, upper=2.0)
    purchase_drop = _series(frame, 'purchase_change_rate_14d').clip(lower=-1.0, upper=2.0)
    support_pressure = normalize(_series(frame, 'support_contact_30d')).clip(0.0, 1.0)

    hazard = (
        0.9 * churn_prob
        + 0.25 * normalize(inactivity).clip(0.0, 1.0)
        + 0.20 * normalize(purchase_anomaly).clip(0.0, 1.0)
        + 0.12 * (-visit_drop).clip(lower=0.0)
        + 0.12 * (-purchase_drop).clip(lower=0.0)
        + 0.08 * support_pressure
    ).clip(lower=0.01)

    median_days = np.clip(np.round(horizon_days * (1.15 - hazard)), 1, horizon_days)
    median_days = pd.Series(median_days, index=frame.index, dtype=float)
    scale = median_days / np.log(2.0)
    survival_14d = np.exp(-14.0 / np.maximum(scale, 1.0))
    survival_30d = np.exp(-30.0 / np.maximum(scale, 1.0))
    survival_60d = np.exp(-60.0 / np.maximum(scale, 1.0))

    risk_percentile = pd.Series(churn_prob.rank(pct=True, method='average').to_numpy(), index=frame.index, dtype=float)
    risk_group = pd.cut(risk_percentile, bins=[-np.inf, 0.33, 0.66, np.inf], labels=['Low risk', 'Mid risk', 'High risk']).astype(str)

    return pd.DataFrame(
        {
            'customer_id': pd.to_numeric(frame['customer_id'], errors='coerce').astype(int),
            'predicted_hazard_ratio': hazard.round(6),
            'predicted_median_time_to_churn_days': median_days.round(0).astype(int),
            'risk_percentile': risk_percentile.round(6),
            'risk_group': risk_group,
            'survival_prob_14d': np.clip(survival_14d, 0.0, 1.0),
            'survival_prob_30d': np.clip(survival_30d, 0.0, 1.0),
            'survival_prob_60d': np.clip(survival_60d, 0.0, 1.0),
        }
    )



def assign_customer_segment(frame: pd.DataFrame, uplift_score: pd.Series, predicted_clv: pd.Series) -> tuple[pd.Series, pd.Series]:
    high_value_cut = float(predicted_clv.quantile(0.70)) if len(predicted_clv) else 0.0
    is_high_value = predicted_clv >= high_value_cut

    uplift_segment = np.select(
        [
            uplift_score >= 0.08,
            (uplift_score >= -0.01) & (uplift_score < 0.08),
            uplift_score < -0.01,
        ],
        ['Persuadables', 'Sure Things', 'Lost Causes'],
        default='Sure Things',
    )
    new_customer = _series(frame, 'customer_age_days') < 45
    customer_segment = np.where(
        new_customer,
        'New Customers',
        np.where(
            is_high_value,
            np.where(uplift_segment == 'Persuadables', 'High Value-Persuadables', np.where(uplift_segment == 'Sure Things', 'High Value-Sure Things', 'High Value-Lost Causes')),
            np.where(uplift_segment == 'Persuadables', 'Low Value-Persuadables', np.where(uplift_segment == 'Sure Things', 'Low Value-Sure Things', 'Low Value-Lost Causes')),
        ),
    )
    return pd.Series(customer_segment, index=frame.index, dtype=object), pd.Series(uplift_segment, index=frame.index, dtype=object)



def prepare_engine_frame(
    fresh_features: pd.DataFrame,
    churn_scores: pd.Series,
    *,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = fresh_features.copy()
    frame['customer_id'] = pd.to_numeric(frame['customer_id'], errors='coerce').astype(int)
    churn_aligned = churn_scores.reindex(frame['customer_id']).fillna(churn_scores.mean() if len(churn_scores) else 0.5)
    frame['churn_probability'] = _clip01(churn_aligned.to_numpy())
    frame['scenario_family'] = scenario_family
    frame['decision_date'] = str(pd.Timestamp(decision_date).date())
    frame['discount_pressure_score'] = (
        _series(frame, 'coupon_cost_30d') / 10000.0
        + 0.60 * normalize(_series(frame, 'coupon_open_30d')).clip(0.0, 1.0)
        + 0.50 * normalize(_series(frame, 'support_contact_30d')).clip(0.0, 1.0)
    ).clip(lower=0.0)

    uplift_score = compute_uplift_score(frame, scenario_family=scenario_family)
    predicted_clv = compute_predicted_clv(frame)
    customer_segment, uplift_segment = assign_customer_segment(frame, uplift_score=uplift_score, predicted_clv=predicted_clv)
    survival_predictions = compute_survival_predictions(frame, churn_probability=frame['churn_probability'])

    frame['uplift_score'] = uplift_score.round(6)
    frame['predicted_clv_12m'] = predicted_clv.round(2)
    frame['clv'] = frame['predicted_clv_12m']
    frame['customer_segment'] = customer_segment
    frame['uplift_segment'] = uplift_segment
    frame['base_expected_revenue'] = (predicted_clv * uplift_score.clip(lower=0.0)).round(2)
    frame['expected_incremental_profit'] = (frame['base_expected_revenue'] * (1.0 - 0.12 * normalize(frame['discount_pressure_score']).clip(0.0, 1.0))).round(2)

    strategy_df = pd.DataFrame.from_dict(STRATEGY_BY_SEGMENT, orient='index').reset_index().rename(columns={'index': 'customer_segment'})
    frame = frame.merge(strategy_df, on='customer_segment', how='left')
    frame['strategy_name'] = frame['strategy_name'].fillna('Generic retention offer')
    frame['strategy_cost'] = _series(frame, 'strategy_cost').clip(lower=0.0)
    frame['strategy_effect_multiplier'] = _series(frame, 'strategy_effect_multiplier', default=1.0).clip(lower=0.0)
    return frame, survival_predictions



def _greedy_select(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    if candidates.empty or budget <= 0:
        return candidates.head(0).copy()

    ranked = candidates.copy()
    ranked = ranked[ranked['coupon_cost'] > 0].copy()
    ranked = ranked[ranked['expected_incremental_profit'] > 0].copy()
    ranked = ranked.sort_values(
        ['priority_score', 'expected_incremental_profit', 'expected_roi', 'timing_urgency_score', 'customer_id', 'coupon_cost'],
        ascending=[False, False, False, False, True, True],
    )

    selected_rows: list[dict[str, Any]] = []
    used_customers: set[int] = set()
    spent = 0.0
    for row in ranked.itertuples(index=False):
        cid = int(getattr(row, 'customer_id'))
        cost = float(getattr(row, 'coupon_cost', 0.0))
        if cid in used_customers:
            continue
        if spent + cost > float(budget):
            continue
        used_customers.add(cid)
        spent += cost
        selected_rows.append(row._asdict())
    if not selected_rows:
        return ranked.head(0).copy()
    return pd.DataFrame(selected_rows)



def run_policy_selection(
    *,
    fresh_features: pd.DataFrame,
    churn_scores: pd.Series,
    budget: int,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
    use_learned_dose_response: bool = False,
) -> PolicySelection:
    engine_frame, survival_predictions = prepare_engine_frame(
        fresh_features=fresh_features,
        churn_scores=churn_scores,
        scenario_family=scenario_family,
        decision_date=decision_date,
    )
    candidates = build_intensity_action_candidates(
        engine_frame,
        survival_predictions=survival_predictions,
        use_learned_dose_response=use_learned_dose_response,
    )
    selected = _greedy_select(candidates, budget=budget)
    total_profit = float(selected['expected_incremental_profit'].sum()) if len(selected) else 0.0
    total_spent = float(selected['coupon_cost'].sum()) if len(selected) else 0.0
    summary = {
        'budget': int(budget),
        'num_selected': int(len(selected)),
        'policy_value': round(total_profit, 6),
        'spent': round(total_spent, 6),
        'avg_selected_churn_probability': round(float(selected['churn_probability'].mean()) if len(selected) else 0.0, 6),
        'avg_selected_uplift_score': round(float(selected['uplift_score'].mean()) if len(selected) else 0.0, 6),
        'avg_intervention_window_days': round(float(selected['intervention_window_days'].mean()) if len(selected) else 0.0, 6),
    }
    return PolicySelection(budget=int(budget), candidates=candidates, selected=selected, summary=summary)



def compute_policy_comparison_metrics(
    *,
    fresh_selection: PolicySelection,
    candidate_selection: PolicySelection,
    latency_days: int,
) -> dict[str, Any]:
    fresh_ids = set(pd.to_numeric(fresh_selection.selected.get('customer_id', pd.Series(dtype=float)), errors='coerce').dropna().astype(int).tolist())
    cand_ids = set(pd.to_numeric(candidate_selection.selected.get('customer_id', pd.Series(dtype=float)), errors='coerce').dropna().astype(int).tolist())
    overlap = len(fresh_ids & cand_ids) / max(len(fresh_ids), 1)

    high_risk_cut = float(fresh_selection.candidates['churn_probability'].quantile(0.75)) if len(fresh_selection.candidates) else 1.0
    high_risk_fresh = set(
        pd.to_numeric(
            fresh_selection.selected.loc[fresh_selection.selected['churn_probability'] >= high_risk_cut, 'customer_id'],
            errors='coerce',
        ).dropna().astype(int).tolist()
    ) if len(fresh_selection.selected) else set()
    missed_at_risk = len(high_risk_fresh - cand_ids) / max(len(high_risk_fresh), 1)

    if len(candidate_selection.selected):
        window_miss = float((pd.to_numeric(candidate_selection.selected['intervention_window_days'], errors='coerce').fillna(999) <= int(latency_days)).mean())
    else:
        window_miss = 0.0

    fresh_value = float(fresh_selection.summary['policy_value'])
    cand_value = float(candidate_selection.summary['policy_value'])
    stale_regret = fresh_value - cand_value
    relative_loss = stale_regret / max(abs(fresh_value), 1.0)

    return {
        'policy_value': round(cand_value, 6),
        'fresh_policy_value': round(fresh_value, 6),
        'stale_regret': round(stale_regret, 6),
        'relative_loss': round(relative_loss, 6),
        'target_overlap': round(float(overlap), 6),
        'missed_at_risk': round(float(missed_at_risk), 6),
        'window_miss_rate': round(float(window_miss), 6),
        'selected_customers': int(len(candidate_selection.selected)),
    }





def matched_reoptimization_policy(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    fresh_features: pd.DataFrame,
    budget: int,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
    reopt_ids: set[int],
    use_learned_dose_response: bool = False,
) -> tuple[PolicySelection, dict[str, Any]]:
    stale_aligned = stale_scores.reindex(fresh_scores.index).fillna(stale_scores.mean() if len(stale_scores) else 0.5)
    if not reopt_ids:
        baseline_selection = run_policy_selection(
            fresh_features=fresh_features,
            churn_scores=stale_aligned,
            budget=budget,
            scenario_family=scenario_family,
            decision_date=decision_date,
            use_learned_dose_response=use_learned_dose_response,
        )
        return baseline_selection, {'reoptimized_customers': 0, 'optimization_call_ratio': 0.0}

    refreshed_scores = stale_aligned.copy()
    valid_ids = [cid for cid in reopt_ids if cid in refreshed_scores.index]
    if valid_ids:
        refreshed_scores.loc[valid_ids] = fresh_scores.loc[valid_ids]
    selection = run_policy_selection(
        fresh_features=fresh_features,
        churn_scores=refreshed_scores,
        budget=budget,
        scenario_family=scenario_family,
        decision_date=decision_date,
        use_learned_dose_response=use_learned_dose_response,
    )
    return selection, {
        'reoptimized_customers': int(len(valid_ids)),
        'optimization_call_ratio': round(len(valid_ids) / max(len(fresh_scores), 1), 6),
    }


def select_random_reopt_ids(
    customer_ids: pd.Index | pd.Series | list[int],
    *,
    k: int,
    rng: np.random.Generator,
) -> set[int]:
    ids = pd.Index(pd.to_numeric(pd.Index(customer_ids), errors='coerce').dropna().astype(int).unique())
    if k <= 0 or len(ids) == 0:
        return set()
    k = min(int(k), len(ids))
    chosen = rng.choice(ids.to_numpy(dtype=int), size=k, replace=False)
    return set(pd.Index(chosen).astype(int).tolist())


def select_top_risk_reopt_ids(stale_scores: pd.Series, *, k: int) -> set[int]:
    if k <= 0 or len(stale_scores) == 0:
        return set()
    ranked = stale_scores.sort_values(ascending=False)
    return set(pd.Index(ranked.index[: min(int(k), len(ranked))]).astype(int).tolist())


def select_top_value_reopt_ids(stale_selection: PolicySelection, *, k: int, fallback_scores: pd.Series | None = None) -> set[int]:
    if k <= 0:
        return set()
    customer_value = pd.DataFrame()
    if len(stale_selection.candidates):
        customer_value = stale_selection.candidates[['customer_id', 'expected_incremental_profit']].copy()
        customer_value['customer_id'] = pd.to_numeric(customer_value['customer_id'], errors='coerce').astype('Int64')
        customer_value['expected_incremental_profit'] = pd.to_numeric(customer_value['expected_incremental_profit'], errors='coerce').fillna(0.0)
        customer_value = customer_value.dropna(subset=['customer_id'])
        customer_value = customer_value.groupby('customer_id', as_index=False)['expected_incremental_profit'].max()
        customer_value = customer_value.sort_values(['expected_incremental_profit', 'customer_id'], ascending=[False, True])
    top_ids: list[int] = []
    if not customer_value.empty:
        top_ids = customer_value['customer_id'].astype(int).tolist()[: int(k)]
    if len(top_ids) >= int(k) or fallback_scores is None:
        return set(top_ids)
    ranked = fallback_scores.sort_values(ascending=False)
    for cid in pd.Index(ranked.index).astype(int).tolist():
        if cid not in top_ids:
            top_ids.append(cid)
        if len(top_ids) >= int(k):
            break
    return set(top_ids[: int(k)])



def _aligned_score_delta(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    absolute: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Align stale/fresh churn scores on the fresh customer universe."""
    stale_aligned = stale_scores.reindex(fresh_scores.index).fillna(
        stale_scores.mean() if len(stale_scores) else 0.5,
    )
    delta = (fresh_scores - stale_aligned).fillna(0.0)
    if absolute:
        delta = delta.abs()
    return stale_aligned, delta


def select_partial_reopt_ids(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    score_delta_threshold: float,
    high_risk_threshold: float,
    top_share: float,
) -> set[int]:
    """Stage-1 partial re-optimization selector.

    The paper text defines partial re-optimization by the absolute score
    change |s_fresh - s_stale|. The optional top_share guard keeps the
    refresh set concentrated on the most volatile customers even when the
    fixed theta is loose.
    """
    _, delta = _aligned_score_delta(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        absolute=True,
    )
    if len(delta) == 0:
        return set()
    share = min(max(float(top_share), 0.0), 1.0)
    cutoff = float(delta.quantile(max(0.0, 1.0 - share))) if share > 0.0 else float('inf')
    threshold = max(float(score_delta_threshold), cutoff)
    reopt_mask = (delta >= threshold) | (fresh_scores >= float(high_risk_threshold))
    return set(pd.Index(delta.index[reopt_mask]).astype(int).tolist())


def select_conformal_reopt_ids(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    conformal_q_hat: float,
    high_risk_threshold: float = 0.80,
    exclude_ids: set[int] | None = None,
) -> set[int]:
    """Stage-2 CRC selector for customers not already refreshed."""
    _, delta = _aligned_score_delta(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        absolute=True,
    )
    if len(delta) == 0:
        return set()
    reopt_mask = (delta > float(conformal_q_hat)) | (fresh_scores >= float(high_risk_threshold))
    ids = set(pd.Index(delta.index[reopt_mask]).astype(int).tolist())
    if exclude_ids:
        ids -= set(int(x) for x in exclude_ids)
    return ids

def partial_reoptimization(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    fresh_features: pd.DataFrame,
    stale_selection: PolicySelection,
    budget: int,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
    score_delta_threshold: float,
    high_risk_threshold: float,
    top_share: float,
    use_learned_dose_response: bool = False,
) -> tuple[PolicySelection, dict[str, Any]]:
    stale_aligned, _ = _aligned_score_delta(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        absolute=True,
    )
    reopt_ids = select_partial_reopt_ids(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        score_delta_threshold=score_delta_threshold,
        high_risk_threshold=high_risk_threshold,
        top_share=top_share,
    )

    if not reopt_ids:
        return stale_selection, {
            'reoptimized_customers': 0,
            'optimization_call_ratio': 0.0,
            'regret_recovery_ratio': 0.0,
            'stage1_partial_customers': 0,
        }

    refreshed_scores = stale_aligned.copy()
    valid_ids = [cid for cid in reopt_ids if cid in refreshed_scores.index]
    if valid_ids:
        refreshed_scores.loc[valid_ids] = fresh_scores.loc[valid_ids]
    partial_selection = run_policy_selection(
        fresh_features=fresh_features,
        churn_scores=refreshed_scores,
        budget=budget,
        scenario_family=scenario_family,
        decision_date=decision_date,
        use_learned_dose_response=use_learned_dose_response,
    )
    base_value = float(stale_selection.summary['policy_value'])
    partial_value = float(partial_selection.summary['policy_value'])
    metadata = {
        'reoptimized_customers': int(len(valid_ids)),
        'optimization_call_ratio': round(len(valid_ids) / max(len(fresh_scores), 1), 6),
        'value_gain_vs_stale': round(partial_value - base_value, 6),
        'stage1_partial_customers': int(len(valid_ids)),
    }
    return partial_selection, metadata


# ═══════════════════════════════════════════════════════════════════
#  Conformal Risk Control  &  Uncertainty-based refresh
# ═══════════════════════════════════════════════════════════════════

def compute_conformal_quantile(
    residuals: np.ndarray,
    alpha: float,
) -> float:
    """Split-conformal quantile with finite-sample correction.

    Given calibration residuals R_1 … R_n and miscoverage level α,
    returns q̂ = Quantile(R; ⌈(n+1)(1−α)⌉ / n) which guarantees
    P(|s_fresh − s_stale| ≤ q̂) ≥ 1 − α marginally.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if n == 0:
        return 1.0
    level = min(np.ceil((n + 1) * (1.0 - float(alpha))) / n, 1.0)
    return float(np.quantile(residuals, level))


def conformal_partial_reoptimization(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    fresh_features: pd.DataFrame,
    stale_selection: PolicySelection,
    budget: int,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
    conformal_q_hat: float,
    use_learned_dose_response: bool = False,
) -> tuple[PolicySelection, dict[str, Any]]:
    """Selective re-optimization with conformal-calibrated threshold.

    q̂ is the (1−α) quantile of calibration residuals |s_fresh − s_stale|
    from previous decision weeks.  A customer whose actual score change
    exceeds q̂ is "abnormally volatile" and therefore refreshed.

    α controls the refresh fraction:
      small α  →  high q̂  →  few refreshes  (strict)
      large α  →  low  q̂  →  more refreshes (lenient)

    Unlike a fixed θ, q̂ adapts automatically to the observed score
    volatility in calibration data, eliminating manual threshold tuning.
    """
    stale_aligned = stale_scores.reindex(fresh_scores.index).fillna(
        stale_scores.mean() if len(stale_scores) else 0.5,
    )

    # ── score change magnitude ──
    delta = (fresh_scores - stale_aligned).abs().fillna(0.0)

    # ── conformal threshold: refresh customers whose change exceeds q̂ ──
    exceedance = delta > conformal_q_hat

    # ── also include high-risk customers whose fresh score is extreme ──
    high_risk = fresh_scores >= 0.80
    reopt_mask = exceedance | high_risk
    reopt_ids = set(stale_aligned.index[reopt_mask].astype(int).tolist())

    if not reopt_ids:
        return stale_selection, {
            'reoptimized_customers': 0,
            'optimization_call_ratio': 0.0,
            'conformal_q_hat': round(conformal_q_hat, 6),
            'method': 'conformal',
        }

    refreshed_scores = stale_aligned.copy()
    valid_ids = [cid for cid in reopt_ids if cid in fresh_scores.index]
    if valid_ids:
        refreshed_scores.loc[valid_ids] = fresh_scores.loc[valid_ids]

    selection = run_policy_selection(
        fresh_features=fresh_features,
        churn_scores=refreshed_scores,
        budget=budget,
        scenario_family=scenario_family,
        decision_date=decision_date,
        use_learned_dose_response=use_learned_dose_response,
    )
    return selection, {
        'reoptimized_customers': int(len(valid_ids)),
        'optimization_call_ratio': round(len(valid_ids) / max(len(fresh_scores), 1), 6),
        'conformal_q_hat': round(conformal_q_hat, 6),
        'method': 'conformal',
    }



def hierarchical_partial_crc_reoptimization(
    *,
    stale_scores: pd.Series,
    fresh_scores: pd.Series,
    fresh_features: pd.DataFrame,
    stale_selection: PolicySelection,
    budget: int,
    scenario_family: str,
    decision_date: str | pd.Timestamp,
    score_delta_threshold: float,
    high_risk_threshold: float,
    top_share: float,
    conformal_q_hat: float,
    use_learned_dose_response: bool = False,
) -> tuple[PolicySelection, dict[str, Any]]:
    """Two-stage hierarchical refresh: partial re-optimization then CRC.

    Stage 1 refreshes the low-cost partial re-optimization set. Stage 2
    adds CRC-flagged customers among those not already refreshed, using the
    conformal quantile learned from previous decision weeks. The final
    policy is optimized once with the union of both refresh sets.
    """
    stale_aligned, _ = _aligned_score_delta(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        absolute=True,
    )
    stage1_ids = select_partial_reopt_ids(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        score_delta_threshold=score_delta_threshold,
        high_risk_threshold=high_risk_threshold,
        top_share=top_share,
    )
    stage2_ids = select_conformal_reopt_ids(
        stale_scores=stale_scores,
        fresh_scores=fresh_scores,
        conformal_q_hat=conformal_q_hat,
        high_risk_threshold=high_risk_threshold,
        exclude_ids=stage1_ids,
    )
    total_ids = set(stage1_ids) | set(stage2_ids)

    if not total_ids:
        return stale_selection, {
            'method': 'hierarchical_partial_crc',
            'stage1_partial_customers': 0,
            'stage2_crc_additional_customers': 0,
            'reoptimized_customers': 0,
            'stage1_partial_call_ratio': 0.0,
            'stage2_crc_additional_call_ratio': 0.0,
            'optimization_call_ratio': 0.0,
            'conformal_q_hat': round(float(conformal_q_hat), 6),
        }

    refreshed_scores = stale_aligned.copy()
    valid_total_ids = [cid for cid in total_ids if cid in refreshed_scores.index]
    if valid_total_ids:
        refreshed_scores.loc[valid_total_ids] = fresh_scores.loc[valid_total_ids]

    selection = run_policy_selection(
        fresh_features=fresh_features,
        churn_scores=refreshed_scores,
        budget=budget,
        scenario_family=scenario_family,
        decision_date=decision_date,
        use_learned_dose_response=use_learned_dose_response,
    )
    n = max(len(fresh_scores), 1)
    stage1_valid = [cid for cid in stage1_ids if cid in fresh_scores.index]
    stage2_valid = [cid for cid in stage2_ids if cid in fresh_scores.index]
    return selection, {
        'method': 'hierarchical_partial_crc',
        'stage1_partial_customers': int(len(stage1_valid)),
        'stage2_crc_additional_customers': int(len(stage2_valid)),
        'reoptimized_customers': int(len(valid_total_ids)),
        'stage1_partial_call_ratio': round(len(stage1_valid) / n, 6),
        'stage2_crc_additional_call_ratio': round(len(stage2_valid) / n, 6),
        'optimization_call_ratio': round(len(valid_total_ids) / n, 6),
        'conformal_q_hat': round(float(conformal_q_hat), 6),
    }


def select_uncertainty_reopt_ids(
    ensemble_scores: list[pd.Series],
    *,
    k: int,
) -> set[int]:
    """Select the *k* customers with highest epistemic uncertainty.

    Epistemic uncertainty is estimated as the variance of churn
    predictions across an ensemble of models trained with different
    random seeds.
    """
    if k <= 0 or not ensemble_scores:
        return set()
    aligned = pd.DataFrame(
        {f'_m{i}': s for i, s in enumerate(ensemble_scores)},
    )
    variance = aligned.var(axis=1).fillna(0.0)
    k = min(int(k), len(variance))
    top_idx = variance.nlargest(k).index
    return set(pd.Index(top_idx).astype(int).tolist())

# A13 hierarchical update with call-budget defenses. Overrides earlier simple-union version.
def _rank_hierarchical_refresh_ids(*, stage1_ids: set[int], stage2_ids: set[int], delta: pd.Series, fresh_scores: pd.Series, score_delta_threshold: float, conformal_q_hat: float, high_risk_threshold: float) -> list[int]:
    union_ids = set(int(x) for x in stage1_ids) | set(int(x) for x in stage2_ids)
    if not union_ids:
        return []
    eps = 1e-12
    rows: list[tuple[float, float, float, int, int]] = []
    for cid in union_ids:
        movement = float(delta.get(cid, 0.0))
        fresh = float(fresh_scores.get(cid, 0.0))
        theta_margin = movement / max(float(score_delta_threshold), eps)
        crc_margin = movement / max(float(conformal_q_hat), eps)
        high_risk_bonus = 0.25 if fresh >= float(high_risk_threshold) else 0.0
        crc_tiebreak = 0.05 if cid in stage2_ids else 0.0
        priority = max(theta_margin, crc_margin) + high_risk_bonus + crc_tiebreak
        rows.append((priority, movement, fresh, -int(cid), int(cid)))
    rows.sort(reverse=True)
    return [cid for *_unused, cid in rows]


def hierarchical_partial_crc_reoptimization(*, stale_scores: pd.Series, fresh_scores: pd.Series, fresh_features: pd.DataFrame, stale_selection: PolicySelection, budget: int, scenario_family: str, decision_date: str | pd.Timestamp, score_delta_threshold: float, high_risk_threshold: float, top_share: float, conformal_q_hat: float, cap_mode: str = 'union', max_call_ratio: float | None = None, use_learned_dose_response: bool = False) -> tuple[PolicySelection, dict[str, Any]]:
    """Two-stage Partial→CRC refresh with explicit call-cost controls.

    cap_mode='union': Partial ∪ CRC upper-bound arm.
    cap_mode='partial_cap': same-cost arm capped at Partial's call count.
    cap_mode='fixed_cap': operational cap of floor(max_call_ratio × N).
    """
    stale_aligned, delta = _aligned_score_delta(stale_scores=stale_scores, fresh_scores=fresh_scores, absolute=True)
    n = max(len(fresh_scores), 1)
    stage1_ids = select_partial_reopt_ids(stale_scores=stale_scores, fresh_scores=fresh_scores, score_delta_threshold=score_delta_threshold, high_risk_threshold=high_risk_threshold, top_share=top_share)
    stage2_candidate_ids = select_conformal_reopt_ids(stale_scores=stale_scores, fresh_scores=fresh_scores, conformal_q_hat=conformal_q_hat, high_risk_threshold=high_risk_threshold, exclude_ids=stage1_ids)
    union_ids = set(stage1_ids) | set(stage2_candidate_ids)
    cap_mode = str(cap_mode or 'union')
    if cap_mode == 'union':
        budget_cap_customers = len(union_ids)
        final_ids = set(union_ids)
    elif cap_mode == 'partial_cap':
        budget_cap_customers = len(stage1_ids)
        ranked = _rank_hierarchical_refresh_ids(stage1_ids=stage1_ids, stage2_ids=stage2_candidate_ids, delta=delta, fresh_scores=fresh_scores, score_delta_threshold=score_delta_threshold, conformal_q_hat=conformal_q_hat, high_risk_threshold=high_risk_threshold)
        final_ids = set(ranked[:max(0, min(int(budget_cap_customers), len(ranked)))])
    elif cap_mode == 'fixed_cap':
        budget_cap_customers = len(stage1_ids) if max_call_ratio is None else int(np.floor(max(0.0, min(float(max_call_ratio), 1.0)) * n))
        ranked = _rank_hierarchical_refresh_ids(stage1_ids=stage1_ids, stage2_ids=stage2_candidate_ids, delta=delta, fresh_scores=fresh_scores, score_delta_threshold=score_delta_threshold, conformal_q_hat=conformal_q_hat, high_risk_threshold=high_risk_threshold)
        final_ids = set(ranked[:max(0, min(int(budget_cap_customers), len(ranked)))])
    else:
        raise ValueError(f'Unsupported hierarchical cap_mode: {cap_mode}')
    n_float = float(n)
    if not final_ids:
        return stale_selection, {'method':'hierarchical_partial_crc','cap_mode':cap_mode,'stage1_partial_customers':int(len(stage1_ids)),'stage2_crc_candidate_customers':int(len(stage2_candidate_ids)),'stage1_partial_retained_customers':0,'stage2_crc_retained_customers':0,'stage2_crc_additional_customers':0,'dropped_union_customers':int(len(union_ids)),'budget_cap_customers':int(budget_cap_customers),'reoptimized_customers':0,'stage1_partial_call_ratio':round(len(stage1_ids)/n_float,6),'stage2_crc_candidate_call_ratio':round(len(stage2_candidate_ids)/n_float,6),'stage2_crc_additional_call_ratio':0.0,'optimization_call_ratio':0.0,'conformal_q_hat':round(float(conformal_q_hat),6)}
    refreshed_scores = stale_aligned.copy()
    valid_total_ids = [cid for cid in final_ids if cid in refreshed_scores.index]
    if valid_total_ids:
        refreshed_scores.loc[valid_total_ids] = fresh_scores.loc[valid_total_ids]
    selection = run_policy_selection(fresh_features=fresh_features, churn_scores=refreshed_scores, budget=budget, scenario_family=scenario_family, decision_date=decision_date, use_learned_dose_response=use_learned_dose_response)
    final_set = set(valid_total_ids)
    stage1_retained = final_set & set(stage1_ids)
    stage2_retained = final_set & set(stage2_candidate_ids)
    return selection, {'method':'hierarchical_partial_crc','cap_mode':cap_mode,'stage1_partial_customers':int(len(stage1_ids)),'stage2_crc_candidate_customers':int(len(stage2_candidate_ids)),'stage1_partial_retained_customers':int(len(stage1_retained)),'stage2_crc_retained_customers':int(len(stage2_retained)),'stage2_crc_additional_customers':int(len(stage2_retained)),'dropped_union_customers':int(len(union_ids-final_set)),'budget_cap_customers':int(budget_cap_customers),'reoptimized_customers':int(len(valid_total_ids)),'stage1_partial_call_ratio':round(len(stage1_ids)/n_float,6),'stage2_crc_candidate_call_ratio':round(len(stage2_candidate_ids)/n_float,6),'stage2_crc_additional_call_ratio':round(len(stage2_retained)/n_float,6),'optimization_call_ratio':round(len(valid_total_ids)/n_float,6),'conformal_q_hat':round(float(conformal_q_hat),6)}

