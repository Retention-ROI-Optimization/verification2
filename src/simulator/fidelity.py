from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimulationFidelityArtifacts:
    summary_path: str
    markdown_path: str


def _safe_rate(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if float(denom) != 0 else 0.0


def _std_mean_diff(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or 'treatment_group' not in df.columns:
        return 0.0
    tmp = df[[column, 'treatment_group']].copy()
    tmp[column] = pd.to_numeric(tmp[column], errors='coerce')
    tr = tmp.loc[tmp['treatment_group'] == 'treatment', column].dropna()
    ct = tmp.loc[tmp['treatment_group'] == 'control', column].dropna()
    if tr.empty or ct.empty:
        return 0.0
    pooled = np.sqrt((tr.var(ddof=1) + ct.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-9:
        return 0.0
    return float((tr.mean() - ct.mean()) / pooled)


def run_simulation_fidelity_audit(data_dir: Path, result_dir: Path) -> SimulationFidelityArtifacts:
    customers = pd.read_csv(data_dir / 'customers.csv')
    assignments = pd.read_csv(data_dir / 'treatment_assignments.csv')
    events = pd.read_csv(data_dir / 'events.csv')
    orders = pd.read_csv(data_dir / 'orders.csv')
    exposures = pd.read_csv(data_dir / 'campaign_exposures.csv')
    summary = pd.read_csv(data_dir / 'customer_summary.csv')

    joined = customers.merge(assignments[['customer_id', 'treatment_group']], on='customer_id', how='left')

    visit_users = int(events.loc[events['event_type'] == 'visit', 'customer_id'].nunique()) if not events.empty else 0
    browse_users = int(events.loc[events['event_type'].isin(['browse', 'search']), 'customer_id'].nunique()) if not events.empty else 0
    cart_users = int(events.loc[events['event_type'] == 'add_to_cart', 'customer_id'].nunique()) if not events.empty else 0
    purchase_users = int(orders['customer_id'].nunique()) if not orders.empty else 0

    funnel = {
        'visit_user_rate': round(_safe_rate(visit_users, len(customers)), 6),
        'browse_user_rate': round(_safe_rate(browse_users, len(customers)), 6),
        'cart_user_rate': round(_safe_rate(cart_users, len(customers)), 6),
        'purchase_user_rate': round(_safe_rate(purchase_users, len(customers)), 6),
        'cart_to_purchase_user_rate': round(_safe_rate(purchase_users, max(cart_users, 1)), 6),
    }

    balance_cols = ['price_sensitivity', 'coupon_affinity', 'treatment_lift_base', 'support_contact_propensity']
    balance = {col: round(_std_mean_diff(joined, col), 6) for col in balance_cols if col in joined.columns}
    max_abs_smd = max((abs(v) for v in balance.values()), default=0.0)

    summary_numeric = summary.copy()
    for col in ['churn_probability', 'inactivity_days', 'coupon_exposure_count', 'coupon_redeem_count', 'coupon_fatigue_score']:
        if col in summary_numeric.columns:
            summary_numeric[col] = pd.to_numeric(summary_numeric[col], errors='coerce')

    churn_corr = float(summary_numeric[['churn_probability', 'inactivity_days']].corr(method='spearman').iloc[0, 1]) if {'churn_probability', 'inactivity_days'}.issubset(summary_numeric.columns) else 0.0
    exposure_redeem_corr = float(summary_numeric[['coupon_exposure_count', 'coupon_redeem_count']].corr(method='spearman').iloc[0, 1]) if {'coupon_exposure_count', 'coupon_redeem_count'}.issubset(summary_numeric.columns) else 0.0

    top_decile_threshold = float(summary_numeric['churn_probability'].quantile(0.90)) if 'churn_probability' in summary_numeric.columns else 1.0
    top_decile = summary_numeric[summary_numeric.get('churn_probability', pd.Series(dtype=float)) >= top_decile_threshold].copy() if 'churn_probability' in summary_numeric.columns else summary_numeric.head(0)
    top_decile_actual_churn = float((pd.to_numeric(top_decile.get('inactivity_days', 0), errors='coerce').fillna(0) >= 30).mean()) if len(top_decile) else 0.0
    overall_actual_churn = float((pd.to_numeric(summary_numeric.get('inactivity_days', 0), errors='coerce').fillna(0) >= 30).mean()) if len(summary_numeric) else 0.0

    warnings = []
    if max_abs_smd > 0.10:
        warnings.append('treatment/control imbalance exceeds common SMD guardrail (0.10)')
    if funnel['purchase_user_rate'] > funnel['visit_user_rate']:
        warnings.append('purchase user rate is higher than visit user rate, check event generation')
    if churn_corr < 0.25:
        warnings.append('churn_probability is weakly aligned with inactivity_days')
    if exposure_redeem_corr < 0.10:
        warnings.append('coupon exposure and redeem counts show weak monotonic relation')

    payload = {
        'coverage': {
            'customers': int(len(customers)),
            'events': int(len(events)),
            'orders': int(len(orders)),
            'campaign_exposures': int(len(exposures)),
            'events_per_customer': round(_safe_rate(len(events), len(customers)), 4),
            'orders_per_customer': round(_safe_rate(len(orders), len(customers)), 4),
        },
        'funnel': funnel,
        'treatment_balance': {
            'standardized_mean_difference': balance,
            'max_abs_smd': round(max_abs_smd, 6),
        },
        'behavior_alignment': {
            'spearman_corr_churn_vs_inactivity': round(churn_corr, 6),
            'spearman_corr_exposure_vs_redeem': round(exposure_redeem_corr, 6),
            'actual_churn_share_overall': round(overall_actual_churn, 6),
            'actual_churn_share_top_risk_decile': round(top_decile_actual_churn, 6),
        },
        'discount_pressure': {
            'avg_coupon_exposures': round(float(pd.to_numeric(summary.get('coupon_exposure_count', 0), errors='coerce').fillna(0).mean()), 6) if len(summary) else 0.0,
            'avg_coupon_fatigue_score': round(float(pd.to_numeric(summary.get('coupon_fatigue_score', 0), errors='coerce').fillna(0).mean()), 6) if 'coupon_fatigue_score' in summary.columns else 0.0,
        },
        'warnings': warnings,
    }

    md = [
        '# Simulation fidelity audit',
        '',
        f"- Customers: **{payload['coverage']['customers']:,}**",
        f"- Events: **{payload['coverage']['events']:,}**",
        f"- Orders: **{payload['coverage']['orders']:,}**",
        f"- Max |SMD|: **{payload['treatment_balance']['max_abs_smd']:.3f}**",
        f"- Churn vs inactivity Spearman: **{payload['behavior_alignment']['spearman_corr_churn_vs_inactivity']:.3f}**",
        '',
        '## Warnings',
    ]
    if warnings:
        md.extend([f'- {item}' for item in warnings])
    else:
        md.append('- No major structural warning triggered.')

    summary_path = result_dir / 'simulation_fidelity_summary.json'
    markdown_path = result_dir / 'simulation_fidelity_report.md'
    summary_path = result_dir / 'simulation_fidelity_summary.json'
    markdown_path = result_dir / 'simulation_fidelity_report.md'
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    markdown_path.write_text('\n'.join(md), encoding='utf-8')
    return SimulationFidelityArtifacts(str(summary_path), str(markdown_path))
