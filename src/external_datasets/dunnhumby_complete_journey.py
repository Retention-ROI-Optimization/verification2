from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CAMPAIGN_COST_BY_TYPE = {
    'TypeA': 3000,
    'TypeB': 7000,
    'TypeC': 12000,
}


@dataclass(frozen=True)
class ImportConfig:
    project_root: Path
    source_path: Path
    seeds: tuple[int, ...] = (41,)
    household_limit: int | None = None
    snapshot_frequency_days: int = 7
    churn_inactivity_days: int = 30
    dormant_inactivity_days: int = 14
    start_date: str = '2023-01-01'


REQUIRED_TABLES = {
    'transaction_data.csv',
    'hh_demographic.csv',
    'campaign_desc.csv',
    'campaign_table.csv',
    'coupon_redempt.csv',
    'product.csv',
}


def _safe_ratio(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray, *, default: float = 0.0) -> np.ndarray:
    num = np.asarray(numer, dtype=float)
    den = np.asarray(denom, dtype=float)
    out = np.full_like(num, float(default), dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    out[~np.isfinite(out)] = float(default)
    return out


def _parse_time_hhmm(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors='coerce').fillna(1200).astype(int)
    hh = (vals // 100).clip(lower=0, upper=23)
    mm = (vals % 100).clip(lower=0, upper=59)
    return pd.to_timedelta(hh, unit='h') + pd.to_timedelta(mm, unit='m')


def _coerce_source_dir(source_path: Path, work_dir: Path) -> Path:
    if source_path.is_dir():
        return source_path
    if source_path.suffix.lower() != '.zip':
        raise ValueError(f'Expected zip or directory, got: {source_path}')
    extract_root = work_dir / 'extracted_complete_journey'
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path) as zf:
        needed = [name for name in zf.namelist() if Path(name).name in REQUIRED_TABLES]
        if len(needed) < len(REQUIRED_TABLES):
            missing = sorted(REQUIRED_TABLES - {Path(name).name for name in needed})
            raise ValueError(f'Missing required tables in archive: {missing}')
        for name in needed:
            target = extract_root / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    return extract_root


def _read_inputs(source_dir: Path) -> dict[str, pd.DataFrame]:
    tx = pd.read_csv(
        source_dir / 'transaction_data.csv',
        dtype={
            'household_key': 'int32',
            'BASKET_ID': 'int64',
            'DAY': 'int32',
            'PRODUCT_ID': 'int32',
            'QUANTITY': 'int16',
            'SALES_VALUE': 'float32',
            'STORE_ID': 'int32',
            'RETAIL_DISC': 'float32',
            'TRANS_TIME': 'int32',
            'WEEK_NO': 'int16',
            'COUPON_DISC': 'float32',
            'COUPON_MATCH_DISC': 'float32',
        },
    )
    demo = pd.read_csv(source_dir / 'hh_demographic.csv')
    campaign_desc = pd.read_csv(source_dir / 'campaign_desc.csv')
    campaign_table = pd.read_csv(source_dir / 'campaign_table.csv')
    coupon_redempt = pd.read_csv(source_dir / 'coupon_redempt.csv')
    product = pd.read_csv(source_dir / 'product.csv', usecols=['PRODUCT_ID', 'BRAND', 'DEPARTMENT', 'COMMODITY_DESC'])
    return {
        'transactions': tx,
        'demographic': demo,
        'campaign_desc': campaign_desc,
        'campaign_table': campaign_table,
        'coupon_redempt': coupon_redempt,
        'product': product,
    }


def _select_households(tx: pd.DataFrame, limit: int | None) -> np.ndarray:
    households = np.sort(tx['household_key'].dropna().astype(int).unique())
    if limit is None or limit >= len(households):
        return households
    rng = np.random.default_rng(42)
    chosen = np.sort(rng.choice(households, size=int(limit), replace=False))
    return chosen


def _prepare_transactions(tx: pd.DataFrame, product: pd.DataFrame, households: np.ndarray, start_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = tx.loc[tx['household_key'].isin(households)].copy()
    tx['date'] = start_date + pd.to_timedelta(tx['DAY'] - 1, unit='D')
    tx['timestamp'] = tx['date'] + _parse_time_hhmm(tx['TRANS_TIME'])
    tx = tx.merge(product, on='PRODUCT_ID', how='left')
    tx['total_discount'] = -(tx['RETAIL_DISC'].fillna(0.0) + tx['COUPON_DISC'].fillna(0.0) + tx['COUPON_MATCH_DISC'].fillna(0.0))
    tx['coupon_discount'] = -(tx['COUPON_DISC'].fillna(0.0) + tx['COUPON_MATCH_DISC'].fillna(0.0))

    basket_level = tx.groupby(['household_key', 'BASKET_ID'], as_index=False).agg(
        order_time=('timestamp', 'max'),
        order_date=('date', 'max'),
        quantity=('QUANTITY', 'sum'),
        net_amount=('SALES_VALUE', 'sum'),
        discount_amount=('total_discount', 'sum'),
        coupon_discount=('coupon_discount', 'sum'),
        store_id=('STORE_ID', 'last'),
        item_category=('DEPARTMENT', lambda s: s.mode().iloc[0] if not s.mode().empty else 'UNKNOWN'),
        unique_products=('PRODUCT_ID', 'nunique'),
    )
    basket_level['gross_amount'] = basket_level['net_amount'] + basket_level['discount_amount']
    basket_level['coupon_used'] = (basket_level['coupon_discount'] > 0).astype(int)
    basket_level['customer_id'] = basket_level['household_key'].astype(int)
    basket_level['order_id'] = basket_level['BASKET_ID'].astype(str)
    orders = basket_level[['order_id', 'customer_id', 'order_time', 'item_category', 'quantity', 'gross_amount', 'discount_amount', 'net_amount', 'coupon_used']].copy()
    return tx, orders


def _build_campaign_tables(
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
    households: np.ndarray,
    start_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    campaign_desc = campaign_desc.copy()
    campaign_desc['coupon_cost'] = campaign_desc['DESCRIPTION'].map(CAMPAIGN_COST_BY_TYPE).fillna(5000).astype(int)
    exposed = campaign_table.loc[campaign_table['household_key'].isin(households)].copy()
    exposed = exposed.merge(campaign_desc[['CAMPAIGN', 'START_DAY', 'END_DAY', 'coupon_cost']], on='CAMPAIGN', how='left')
    exposed['exposure_time'] = start_date + pd.to_timedelta(exposed['START_DAY'] - 1, unit='D') + pd.to_timedelta(9, unit='h')
    exposed['customer_id'] = exposed['household_key'].astype(int)
    exposed['campaign_type'] = exposed['DESCRIPTION'].fillna('TypeA')
    exposed['coupon_cost'] = exposed['coupon_cost'].fillna(5000).astype(int)
    exposed['exposure_id'] = (
        exposed['customer_id'].astype(str) + '_camp_' + exposed['CAMPAIGN'].astype(str)
    )
    exposures = exposed[['exposure_id', 'customer_id', 'exposure_time', 'campaign_type', 'coupon_cost', 'CAMPAIGN', 'END_DAY']].copy()

    redempt = coupon_redempt.loc[coupon_redempt['household_key'].isin(households)].copy()
    redeem_counts = redempt.groupby('household_key').size().rename('coupon_redeem_count_total')
    campaign_redeem = redempt.groupby(['household_key', 'CAMPAIGN']).size().rename('campaign_redeem_count').reset_index()
    exposures = exposures.merge(campaign_redeem, left_on=['customer_id', 'CAMPAIGN'], right_on=['household_key', 'CAMPAIGN'], how='left')
    exposures['campaign_redeem_count'] = exposures['campaign_redeem_count'].fillna(0).astype(int)
    exposures = exposures.drop(columns=['household_key'])

    first_exposure = exposures.sort_values(['customer_id', 'exposure_time']).groupby('customer_id', as_index=False).agg(
        assigned_at=('exposure_time', 'first'),
        campaign_type=('campaign_type', 'first'),
        coupon_cost=('coupon_cost', 'mean'),
        treatment_flag=('customer_id', lambda s: 1),
        exposure_count=('customer_id', 'size'),
        redeemed_campaigns=('campaign_redeem_count', lambda s: int((s > 0).sum())),
    )
    first_exposure['treatment_group'] = 'contacted'
    first_exposure['coupon_cost'] = first_exposure['coupon_cost'].round().astype(int)
    return exposures[['exposure_id', 'customer_id', 'exposure_time', 'campaign_type', 'coupon_cost']], first_exposure.merge(redeem_counts, left_on='customer_id', right_index=True, how='left')


def _build_customers(
    tx: pd.DataFrame,
    orders: pd.DataFrame,
    demo: pd.DataFrame,
    treatment_contacted: pd.DataFrame,
) -> pd.DataFrame:
    tx = tx.copy()
    spend = tx.groupby('household_key').agg(
        total_sales=('SALES_VALUE', 'sum'),
        total_discount=('total_discount', 'sum'),
        coupon_discount=('coupon_discount', 'sum'),
        national_brand_share=('BRAND', lambda s: float((s.fillna('Unknown').astype(str).str.lower() == 'national').mean())),
        store_id_mode=('STORE_ID', lambda s: s.mode().iloc[0] if not s.mode().empty else int(s.iloc[0])),
    )
    order_stats = orders.groupby('customer_id').agg(
        signup_date=('order_time', 'min'),
        last_purchase_date=('order_time', 'max'),
        frequency=('order_id', 'count'),
        monetary=('net_amount', 'sum'),
        coupon_order_ratio=('coupon_used', 'mean'),
        avg_order_value=('net_amount', 'mean'),
    )
    gaps = orders[['customer_id', 'order_time']].sort_values(['customer_id', 'order_time']).copy()
    gaps['prev_time'] = gaps.groupby('customer_id')['order_time'].shift(1)
    gaps['gap_days'] = (gaps['order_time'] - gaps['prev_time']).dt.total_seconds() / 86400.0
    gap_stats = gaps.groupby('customer_id').agg(avg_gap_days=('gap_days', 'mean')).fillna({'avg_gap_days': 999.0})

    demo = demo.rename(columns={'household_key': 'customer_id'}).copy()
    base = order_stats.join(spend, how='left')
    base.index = base.index.astype(int)
    base = base.join(gap_stats, how='left')
    base = base.join(treatment_contacted.set_index('customer_id')[['exposure_count', 'redeemed_campaigns', 'coupon_redeem_count_total']], how='left')
    base[['exposure_count', 'redeemed_campaigns', 'coupon_redeem_count_total']] = base[['exposure_count', 'redeemed_campaigns', 'coupon_redeem_count_total']].fillna(0)
    base = base.reset_index().rename(columns={'index': 'customer_id'})
    base = base.merge(demo, on='customer_id', how='left')

    total_sales = base['total_sales'].replace(0, np.nan)
    discount_share = pd.Series(_safe_ratio(base['total_discount'], total_sales, default=0.0), index=base.index).clip(0.0, 1.0)
    coupon_share = pd.Series(_safe_ratio(base['coupon_discount'], total_sales, default=0.0), index=base.index).clip(0.0, 1.0)
    coupon_affinity = (0.55 * base['coupon_order_ratio'].fillna(0.0) + 0.30 * coupon_share + 0.15 * pd.Series(_safe_ratio(base['redeemed_campaigns'], np.maximum(base['exposure_count'], 1), default=0.0), index=base.index)).clip(0.0, 1.0)
    price_sensitivity = (0.65 * discount_share + 0.35 * coupon_share).clip(0.0, 1.0)
    discount_fatigue_sensitivity = (0.55 * coupon_affinity + 0.45 * pd.Series(_safe_ratio(base['exposure_count'], base['frequency'].replace(0, np.nan), default=0.0), index=base.index)).clip(0.0, 1.0)
    offer_dependency_risk = (0.70 * base['coupon_order_ratio'].fillna(0.0) + 0.30 * coupon_share).clip(0.0, 1.0)
    brand_sensitivity = base['national_brand_share'].fillna(0.5).clip(0.0, 1.0)

    now = base['last_purchase_date'].max()
    recency_days = (now - base['last_purchase_date']).dt.days.clip(lower=0)
    freq_q = base['frequency'].rank(pct=True, method='average')
    spend_q = base['monetary'].rank(pct=True, method='average')
    recency_q = recency_days.rank(pct=True, method='average')

    persona = np.select(
        [
            (freq_q >= 0.8) & (spend_q >= 0.8) & (recency_q <= 0.35),
            (freq_q >= 0.55) & (spend_q >= 0.45) & (recency_q <= 0.55),
            price_sensitivity >= 0.60,
            (freq_q <= 0.25) & (recency_q >= 0.75),
            base['avg_gap_days'].fillna(999.0) > 35,
        ],
        ['vip_loyal', 'regular_loyal', 'price_sensitive', 'churn_progressing', 'explorer'],
        default='regular_loyal',
    )

    treatment_lift_base = (
        0.16 * coupon_affinity
        + 0.10 * pd.Series(_safe_ratio(base['redeemed_campaigns'], np.maximum(base['exposure_count'], 1), default=0.0), index=base.index)
        - 0.10 * price_sensitivity
        - 0.06 * brand_sensitivity
    ).clip(-0.15, 0.35)
    uplift_segment_true = np.select(
        [
            treatment_lift_base >= 0.12,
            (treatment_lift_base >= 0.03) & (treatment_lift_base < 0.12),
            (treatment_lift_base > -0.03) & (treatment_lift_base < 0.03),
        ],
        ['persuadable', 'sure_thing', 'lost_cause'],
        default='sleeping_dog',
    )

    region = 'store_cluster_' + (base['store_id_mode'].fillna(0).astype(int) % 5).astype(str)

    customers = pd.DataFrame(
        {
            'customer_id': base['customer_id'].astype(int),
            'persona': persona,
            'uplift_segment_true': uplift_segment_true,
            'signup_date': pd.to_datetime(base['signup_date']).dt.floor('D'),
            'acquisition_month': pd.to_datetime(base['signup_date']).dt.strftime('%Y-%m'),
            'region': region,
            'device_type': 'offline_store',
            'acquisition_channel': 'in_store',
            'coupon_affinity': coupon_affinity.round(6),
            'price_sensitivity': price_sensitivity.round(6),
            'discount_fatigue_sensitivity': discount_fatigue_sensitivity.round(6),
            'offer_dependency_risk': offer_dependency_risk.round(6),
            'brand_sensitivity': brand_sensitivity.round(6),
        }
    )
    return customers, pd.DataFrame(
        {
            'customer_id': base['customer_id'].astype(int),
            'treatment_lift_base': treatment_lift_base.round(6),
        }
    )


def _build_treatment_assignments(customers: pd.DataFrame, contacted: pd.DataFrame, lift: pd.DataFrame) -> pd.DataFrame:
    base = customers[['customer_id', 'signup_date']].copy()
    out = base.merge(contacted, on='customer_id', how='left').merge(lift, on='customer_id', how='left')
    out['assigned_at'] = pd.to_datetime(out['assigned_at']).fillna(pd.to_datetime(out['signup_date']))
    out['campaign_type'] = out['campaign_type'].fillna('none')
    out['coupon_cost'] = pd.to_numeric(out['coupon_cost'], errors='coerce').fillna(0).round().astype(int)
    out['treatment_flag'] = pd.to_numeric(out['treatment_flag'], errors='coerce').fillna(0).astype(int)
    out['treatment_group'] = np.where(out['treatment_flag'].eq(1), 'contacted', 'control')
    out['exposure_count'] = pd.to_numeric(out.get('exposure_count', 0), errors='coerce').fillna(0).astype(int)
    out['redeemed_campaigns'] = pd.to_numeric(out.get('redeemed_campaigns', 0), errors='coerce').fillna(0).astype(int)
    out['coupon_redeem_count_total'] = pd.to_numeric(out.get('coupon_redeem_count_total', 0), errors='coerce').fillna(0).astype(int)
    return out[['customer_id', 'treatment_group', 'treatment_flag', 'campaign_type', 'coupon_cost', 'assigned_at', 'exposure_count', 'redeemed_campaigns', 'coupon_redeem_count_total', 'treatment_lift_base']]


def _build_events(orders: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    seq = orders[['customer_id', 'order_id', 'order_time', 'quantity', 'item_category']].copy()
    seq['session_id'] = 's_' + seq['order_id'].astype(str)
    seq['start_time'] = pd.to_datetime(seq['order_time']) - pd.to_timedelta(np.minimum(seq['quantity'].clip(lower=1), 4) * 4 + 12, unit='m')

    blocks: list[pd.DataFrame] = []
    for event_type, offset_min in [
        ('visit', 0),
        ('page_view', 2),
        ('search', 5),
        ('add_to_cart', 8),
        ('purchase', None),
    ]:
        df = seq[['customer_id', 'session_id', 'item_category']].copy()
        if event_type == 'purchase':
            df['timestamp'] = pd.to_datetime(seq['order_time'])
        else:
            df['timestamp'] = pd.to_datetime(seq['start_time']) + pd.to_timedelta(offset_min, unit='m')
        df['event_type'] = event_type
        if event_type == 'search':
            mask = seq['quantity'].fillna(1).astype(int) >= 2
            df = df.loc[mask.values].copy()
        blocks.append(df)

    if not exposures.empty:
        exp = exposures[['customer_id', 'exposure_time']].copy()
        exp['session_id'] = 'campaign_' + np.arange(len(exp)).astype(str)
        exp['item_category'] = 'campaign'
        exp['timestamp'] = pd.to_datetime(exp['exposure_time']) + pd.to_timedelta(6, unit='h')
        exp['event_type'] = 'coupon_open'
        blocks.append(exp[['customer_id', 'session_id', 'item_category', 'timestamp', 'event_type']])

    order_gaps = orders[['customer_id', 'order_time']].sort_values(['customer_id', 'order_time']).copy()
    order_gaps['prev_time'] = order_gaps.groupby('customer_id')['order_time'].shift(1)
    order_gaps['gap_days'] = (pd.to_datetime(order_gaps['order_time']) - pd.to_datetime(order_gaps['prev_time'])).dt.total_seconds() / 86400.0
    support = order_gaps.loc[order_gaps['gap_days'] >= 45, ['customer_id', 'prev_time', 'order_time']].copy()
    if not support.empty:
        support['timestamp'] = pd.to_datetime(support['prev_time']) + (pd.to_datetime(support['order_time']) - pd.to_datetime(support['prev_time'])) / 2
        support['session_id'] = 'support_' + np.arange(len(support)).astype(str)
        support['item_category'] = 'support'
        support['event_type'] = 'support_contact'
        blocks.append(support[['customer_id', 'session_id', 'item_category', 'timestamp', 'event_type']])

    events = pd.concat(blocks, ignore_index=True)
    events = events.sort_values(['customer_id', 'timestamp', 'event_type']).reset_index(drop=True)
    events['event_id'] = 'evt_' + np.arange(1, len(events) + 1).astype(str)
    return events[['event_id', 'customer_id', 'timestamp', 'event_type', 'session_id', 'item_category']]


def _build_state_snapshots(
    customers: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    snapshot_frequency_days: int,
    dormant_inactivity_days: int,
    churn_inactivity_days: int,
) -> pd.DataFrame:
    start = pd.to_datetime(customers['signup_date']).min().floor('D')
    end = pd.to_datetime(orders['order_time']).max().floor('D')
    snapshot_dates = pd.date_range(start=start, end=end, freq=f'{int(snapshot_frequency_days)}D')

    visit_dates = (
        events.loc[events['event_type'] == 'visit', ['customer_id', 'timestamp']]
        .assign(date=lambda x: pd.to_datetime(x['timestamp']).dt.floor('D'))
        .groupby('customer_id')['date']
        .apply(list)
        .to_dict()
    )
    order_dates = orders.assign(date=pd.to_datetime(orders['order_time']).dt.floor('D')).groupby('customer_id')['date'].apply(list).to_dict()
    order_amounts = orders.assign(date=pd.to_datetime(orders['order_time']).dt.floor('D')).groupby('customer_id').apply(
        lambda df: list(zip(df['date'].tolist(), pd.to_numeric(df['net_amount'], errors='coerce').fillna(0.0).tolist()))
    ).to_dict()
    exposure_dates = exposures.assign(date=pd.to_datetime(exposures['exposure_time']).dt.floor('D')).groupby('customer_id')['date'].apply(list).to_dict()

    rows: list[dict[str, object]] = []
    for customer_id, signup_date in customers[['customer_id', 'signup_date']].itertuples(index=False):
        cid = int(customer_id)
        signup = pd.Timestamp(signup_date).floor('D')
        visits = sorted(visit_dates.get(cid, []))
        purchases = sorted(order_dates.get(cid, []))
        spend_pairs = sorted(order_amounts.get(cid, []), key=lambda x: x[0])
        exposures_c = sorted(exposure_dates.get(cid, []))

        for snapshot_date in snapshot_dates:
            if snapshot_date < signup:
                continue
            visit_hist = [d for d in visits if d <= snapshot_date]
            purchase_hist = [d for d in purchases if d <= snapshot_date]
            exposure_hist = [d for d in exposures_c if d <= snapshot_date]
            amount_hist = [amt for d, amt in spend_pairs if d <= snapshot_date]
            last_visit = visit_hist[-1] if visit_hist else pd.NaT
            last_purchase = purchase_hist[-1] if purchase_hist else pd.NaT
            anchor = last_purchase if pd.notna(last_purchase) else last_visit
            inactivity_days = int((snapshot_date - anchor).days) if pd.notna(anchor) else int((snapshot_date - signup).days)
            if inactivity_days >= churn_inactivity_days:
                status = 'churn_risk'
            elif inactivity_days >= dormant_inactivity_days:
                status = 'dormant'
            else:
                status = 'active'

            visit_28 = sum((snapshot_date - d).days <= 28 for d in visit_hist)
            purchase_28 = sum((snapshot_date - d).days <= 28 for d in purchase_hist)
            exposure_28 = sum((snapshot_date - d).days <= 28 for d in exposure_hist)
            rows.append(
                {
                    'customer_id': cid,
                    'snapshot_date': snapshot_date,
                    'last_visit_date': last_visit,
                    'last_purchase_date': last_purchase,
                    'visits_total': int(len(visit_hist)),
                    'purchases_total': int(len(purchase_hist)),
                    'monetary_total': round(float(np.sum(amount_hist)) if amount_hist else 0.0, 2),
                    'inactivity_days': inactivity_days,
                    'current_status': status,
                    'recent_visit_score': round(min(visit_28 / 8.0, 1.0), 6),
                    'recent_purchase_score': round(min(purchase_28 / 4.0, 1.0), 6),
                    'recent_exposure_score': round(min(exposure_28 / 4.0, 1.0), 6),
                }
            )
    return pd.DataFrame(rows)


def _build_customer_summary(customers: pd.DataFrame, orders: pd.DataFrame, exposures: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    latest = snapshots.sort_values(['customer_id', 'snapshot_date']).groupby('customer_id').tail(1)
    order_agg = orders.groupby('customer_id').agg(
        total_orders=('order_id', 'count'),
        total_net_sales=('net_amount', 'sum'),
        avg_order_value=('net_amount', 'mean'),
        coupon_order_ratio=('coupon_used', 'mean'),
    ).reset_index()
    exposure_agg = exposures.groupby('customer_id').agg(
        total_exposures=('exposure_id', 'count'),
        avg_coupon_cost=('coupon_cost', 'mean'),
    ).reset_index()
    summary = customers.merge(order_agg, on='customer_id', how='left').merge(exposure_agg, on='customer_id', how='left').merge(
        latest[['customer_id', 'snapshot_date', 'current_status', 'inactivity_days', 'recent_visit_score', 'recent_purchase_score', 'recent_exposure_score']],
        on='customer_id',
        how='left',
    )
    return summary.fillna({'total_orders': 0, 'total_net_sales': 0.0, 'avg_order_value': 0.0, 'coupon_order_ratio': 0.0, 'total_exposures': 0, 'avg_coupon_cost': 0.0})


def _build_cohort_retention(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    cohort = customers[['customer_id', 'signup_date']].copy()
    cohort['cohort_month'] = pd.to_datetime(cohort['signup_date']).dt.to_period('M').astype(str)
    ords = orders[['customer_id', 'order_time']].copy()
    ords['order_month'] = pd.to_datetime(ords['order_time']).dt.to_period('M').astype(str)
    merged = cohort.merge(ords, on='customer_id', how='left')
    order_period = pd.PeriodIndex(merged['order_month'], freq='M')
    cohort_period = pd.PeriodIndex(merged['cohort_month'], freq='M')
    merged['cohort_period'] = (order_period.year - cohort_period.year) * 12 + (order_period.month - cohort_period.month)
    valid = merged.loc[merged['cohort_period'].notna() & (merged['cohort_period'] >= 0)].copy()
    retained = valid.groupby(['cohort_month', 'cohort_period'])['customer_id'].nunique().rename('retained_customers').reset_index()
    cohort_size = cohort.groupby('cohort_month')['customer_id'].nunique().rename('cohort_size').reset_index()
    out = retained.merge(cohort_size, on='cohort_month', how='left')
    out['retention_rate'] = _safe_ratio(out['retained_customers'], out['cohort_size'], default=0.0)
    return out.sort_values(['cohort_month', 'cohort_period']).reset_index(drop=True)


def _hardlink_or_copytree(src: Path, dst: Path) -> None:
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except Exception:
        shutil.copytree(src, dst)


def _export_seed(seed_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(seed_dir / f'{name}.csv', index=False)


def import_complete_journey(config: ImportConfig) -> dict[str, object]:
    work_dir = config.project_root / 'artifacts' / 'external_imports' / 'complete_journey'
    work_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _coerce_source_dir(config.source_path, work_dir)
    raw = _read_inputs(source_dir)
    households = _select_households(raw['transactions'], config.household_limit)
    start_date = pd.Timestamp(config.start_date)

    tx, orders = _prepare_transactions(raw['transactions'], raw['product'], households, start_date)
    exposures, contacted = _build_campaign_tables(raw['campaign_table'], raw['campaign_desc'], raw['coupon_redempt'], households, start_date)
    customers, lift = _build_customers(tx, orders, raw['demographic'], contacted)
    treatment = _build_treatment_assignments(customers, contacted, lift)
    events = _build_events(orders, exposures)
    snapshots = _build_state_snapshots(
        customers,
        events,
        orders,
        exposures,
        snapshot_frequency_days=config.snapshot_frequency_days,
        dormant_inactivity_days=config.dormant_inactivity_days,
        churn_inactivity_days=config.churn_inactivity_days,
    )
    customer_summary = _build_customer_summary(customers, orders, exposures, snapshots)
    cohort_retention = _build_cohort_retention(customers, orders)

    tables = {
        'customers': customers.sort_values('customer_id').reset_index(drop=True),
        'events': events.sort_values(['customer_id', 'timestamp']).reset_index(drop=True),
        'orders': orders.sort_values(['customer_id', 'order_time']).reset_index(drop=True),
        'state_snapshots': snapshots.sort_values(['customer_id', 'snapshot_date']).reset_index(drop=True),
        'campaign_exposures': exposures.sort_values(['customer_id', 'exposure_time']).reset_index(drop=True),
        'treatment_assignments': treatment.sort_values('customer_id').reset_index(drop=True),
        'customer_summary': customer_summary.sort_values('customer_id').reset_index(drop=True),
        'cohort_retention': cohort_retention,
    }

    raw_grid_root = config.project_root / 'artifacts' / 'raw_grid'
    primary_seed = int(config.seeds[0])
    primary_dir = raw_grid_root / f'seed_{primary_seed}'
    if primary_dir.exists():
        shutil.rmtree(primary_dir)
    _export_seed(primary_dir, tables)

    for seed in config.seeds[1:]:
        dst = raw_grid_root / f'seed_{int(seed)}'
        if dst.exists():
            shutil.rmtree(dst)
        _hardlink_or_copytree(primary_dir, dst)

    manifest = {
        'source_path': str(config.source_path),
        'source_dir': str(source_dir),
        'seeds': [int(s) for s in config.seeds],
        'household_count': int(customers['customer_id'].nunique()),
        'order_count': int(len(orders)),
        'event_count': int(len(events)),
        'snapshot_count': int(len(snapshots)),
        'date_min': str(pd.to_datetime(orders['order_time']).min().date()),
        'date_max': str(pd.to_datetime(orders['order_time']).max().date()),
        'primary_seed_dir': str(primary_dir),
    }
    pd.Series(manifest, dtype='object').to_json(work_dir / 'import_manifest.json', force_ascii=False, indent=2)
    return manifest
