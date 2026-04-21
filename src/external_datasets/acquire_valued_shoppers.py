from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CAMPAIGN_TYPE_BY_VALUE = {
    0.75: 'TypeA',
    1.00: 'TypeA',
    1.50: 'TypeB',
    2.00: 'TypeB',
    3.00: 'TypeC',
}


@dataclass(frozen=True)
class ImportConfig:
    project_root: Path
    aggregate_path: Path
    train_history_path: Path
    offers_path: Path
    seeds: tuple[int, ...] = (151, 152, 153)
    household_limit: int | None = None
    snapshot_frequency_days: int = 7
    churn_inactivity_days: int = 30
    dormant_inactivity_days: int = 14
    artifacts_dir: str = 'artifacts'


def _safe_ratio(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray, *, default: float = 0.0) -> np.ndarray:
    num = np.asarray(numer, dtype=float)
    den = np.asarray(denom, dtype=float)
    out = np.full_like(num, float(default), dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    out[~np.isfinite(out)] = float(default)
    return out


def _artifact_root(config: ImportConfig) -> Path:
    path = Path(config.artifacts_dir)
    if path.is_absolute():
        return path.resolve()
    return (config.project_root / path).resolve()


def _read_inputs(config: ImportConfig) -> pd.DataFrame:
    agg = pd.read_csv(config.aggregate_path, parse_dates=['offerdate'])
    history = pd.read_csv(config.train_history_path, parse_dates=['offerdate'])
    offers = pd.read_csv(config.offers_path)
    merged = agg.merge(
        history[['id', 'chain', 'market', 'offer', 'repeattrips', 'repeater', 'offerdate']],
        on=['id', 'offerdate'],
        how='left',
    ).merge(
        offers[['offer', 'category', 'company', 'brand', 'offervalue', 'quantity']],
        on='offer',
        how='left',
    )
    if merged['offer'].isna().any():
        raise ValueError('Failed to align aggregate rows with trainHistory/offers; some offer ids are missing.')
    merged['repeater_flag'] = merged['repeater'].astype(str).str.lower().eq('t').astype(int)
    for col in [
        'total_spend_365d',
        'total_qty_365d',
        'num_visits_365d',
        'days_since_last_purchase',
        'spend_in_offer_category_365d',
        'spend_in_offer_company_365d',
        'spend_in_offer_brand_365d',
        'qty_in_offer_category_365d',
        'qty_in_offer_company_365d',
        'qty_in_offer_brand_365d',
        'offervalue',
    ]:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)
    merged['total_spend_365d'] = merged['total_spend_365d'].clip(lower=0.0)
    merged['total_qty_365d'] = merged['total_qty_365d'].clip(lower=0.0)
    merged['num_visits_365d'] = merged['num_visits_365d'].clip(lower=0.0)
    merged['days_since_last_purchase'] = merged['days_since_last_purchase'].replace([np.inf, -np.inf], np.nan).fillna(365.0).clip(lower=0.0, upper=365.0)
    merged = merged.sort_values(['offerdate', 'id']).reset_index(drop=True)
    if config.household_limit is not None and int(config.household_limit) < len(merged):
        merged = merged.sample(n=int(config.household_limit), random_state=42).sort_values(['offerdate', 'id']).reset_index(drop=True)
    return merged


def _normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce').fillna(0.0)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - low) / (high - low)


def _campaign_type(offervalue: float) -> str:
    return CAMPAIGN_TYPE_BY_VALUE.get(float(round(offervalue, 2)), 'TypeB')


def _coupon_cost(offervalue: float) -> int:
    return int(np.clip(round(float(offervalue) * 10_000), 7_500, 30_000))


def _persona(row: pd.Series, spend_rank: float, recency_rank: float, visit_rank: float) -> str:
    if spend_rank >= 0.80 and recency_rank <= 0.35:
        return 'vip_loyal'
    if visit_rank >= 0.60 and recency_rank <= 0.55:
        return 'regular_loyal'
    if float(row['offervalue']) >= 1.50 or float(row['spend_in_offer_category_365d']) > 0:
        return 'price_sensitive'
    if recency_rank >= 0.80:
        return 'churn_progressing'
    return 'explorer'


def _uplift_segment(row: pd.Series, cat_share: float, brand_share: float) -> str:
    repeattrips = float(row['repeattrips'])
    if repeattrips >= 2 and cat_share >= 0.08:
        return 'persuadable'
    if repeattrips >= 1:
        return 'sure_thing'
    if cat_share <= 0.01 and brand_share <= 0.005:
        return 'lost_cause'
    return 'sleeping_dog'


def _build_customers(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = base.copy()
    total_spend = out['total_spend_365d'].replace(0.0, np.nan)
    cat_share = pd.Series(_safe_ratio(out['spend_in_offer_category_365d'], total_spend, default=0.0), index=out.index).clip(0.0, 1.0)
    comp_share = pd.Series(_safe_ratio(out['spend_in_offer_company_365d'], total_spend, default=0.0), index=out.index).clip(0.0, 1.0)
    brand_share = pd.Series(_safe_ratio(out['spend_in_offer_brand_365d'], total_spend, default=0.0), index=out.index).clip(0.0, 1.0)
    recency_rank = _normalize(out['days_since_last_purchase'])
    spend_rank = _normalize(np.log1p(out['total_spend_365d']))
    visit_rank = _normalize(out['num_visits_365d'])
    offer_value_rank = _normalize(out['offervalue'])

    coupon_affinity = (0.45 * cat_share + 0.25 * comp_share + 0.15 * brand_share + 0.15 * offer_value_rank).clip(0.0, 1.0)
    price_sensitivity = (0.50 * offer_value_rank + 0.25 * cat_share + 0.25 * (1.0 - recency_rank)).clip(0.0, 1.0)
    fatigue = (0.55 * coupon_affinity + 0.25 * visit_rank + 0.20 * offer_value_rank).clip(0.0, 1.0)
    dependency = (0.55 * comp_share + 0.45 * brand_share).clip(0.0, 1.0)
    brand_sensitivity = (0.60 * (1.0 - cat_share) + 0.40 * brand_share).clip(0.0, 1.0)

    personas = [
        _persona(row, float(spend_rank.loc[idx]), float(recency_rank.loc[idx]), float(visit_rank.loc[idx]))
        for idx, row in out.iterrows()
    ]
    uplift_segments = [
        _uplift_segment(row, float(cat_share.loc[idx]), float(brand_share.loc[idx]))
        for idx, row in out.iterrows()
    ]

    signup_date = (pd.to_datetime(out['offerdate']) - pd.to_timedelta(365, unit='D')).dt.floor('D')
    customers = pd.DataFrame(
        {
            'customer_id': pd.to_numeric(out['id'], errors='coerce').astype(int),
            'persona': personas,
            'uplift_segment_true': uplift_segments,
            'signup_date': signup_date,
            'acquisition_month': signup_date.dt.to_period('M').astype(str),
            'region': 'market_' + out['market'].fillna(-1).astype(int).astype(str),
            'device_type': 'offline_store',
            'acquisition_channel': 'coupon_offer',
            'coupon_affinity': coupon_affinity.round(6),
            'price_sensitivity': price_sensitivity.round(6),
            'discount_fatigue_sensitivity': fatigue.round(6),
            'offer_dependency_risk': dependency.round(6),
            'brand_sensitivity': brand_sensitivity.round(6),
        }
    )

    lift = (
        0.16 * out['repeater_flag']
        + 0.18 * cat_share
        + 0.10 * comp_share
        + 0.08 * brand_share
        + 0.06 * (1.0 - recency_rank)
        - 0.14 * price_sensitivity
        - 0.10 * fatigue
    ).clip(-0.15, 0.35)

    lift_df = pd.DataFrame(
        {
            'customer_id': pd.to_numeric(out['id'], errors='coerce').astype(int),
            'treatment_lift_base': lift.round(6),
        }
    )
    return customers, lift_df


def _category_label(category_id: int, fallback_prefix: str = 'CATEGORY') -> str:
    return f'{fallback_prefix}_{int(category_id)}'


def _generic_categories(row: pd.Series) -> list[str]:
    base = [
        _category_label(int(row['category']), 'OFFERCAT'),
        f'CHAIN_{int(row["chain"])}',
        f'MARKET_{int(row["market"])}',
        f'COMPANY_{int(row["company"])}',
        f'BRAND_{int(row["brand"])}',
    ]
    return base


def _pseudo_order_count(visits_365d: float, total_spend_365d: float) -> int:
    if total_spend_365d <= 0 or visits_365d <= 0:
        return 0
    return int(np.clip(round(np.sqrt(float(visits_365d))), 2, 12))


def _allocate_integer(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    if parts <= 0:
        return np.zeros(0, dtype=int)
    if total <= 0:
        return np.ones(parts, dtype=int)
    weights = rng.dirichlet(np.ones(parts, dtype=float))
    values = np.floor(weights * total).astype(int)
    remainder = int(total - values.sum())
    if remainder > 0:
        extra_idx = rng.choice(parts, size=remainder, replace=True)
        for idx in extra_idx:
            values[int(idx)] += 1
    values = np.maximum(values, 1)
    adjust = int(values.sum() - total)
    while adjust > 0:
        idxs = np.where(values > 1)[0]
        if len(idxs) == 0:
            break
        take = min(adjust, len(idxs))
        chosen = rng.choice(idxs, size=take, replace=False)
        values[chosen] -= 1
        adjust = int(values.sum() - total)
    return values


def _build_orders(base: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    next_order_id = int(seed) * 1_000_000_000
    for row in base.itertuples(index=False):
        cid = int(getattr(row, 'id'))
        rng = np.random.default_rng(int(seed) * 1_000_003 + cid)
        offerdate = pd.Timestamp(getattr(row, 'offerdate')).floor('D')
        order_count = _pseudo_order_count(getattr(row, 'num_visits_365d'), getattr(row, 'total_spend_365d'))
        if order_count <= 0:
            continue
        last_gap_days = int(np.clip(round(float(getattr(row, 'days_since_last_purchase'))), 1, 365))
        last_purchase = offerdate - pd.Timedelta(days=last_gap_days)
        start = offerdate - pd.Timedelta(days=365)
        if order_count == 1:
            order_dates = [last_purchase]
        else:
            pool_days = max(int((last_purchase - start).days), 1)
            sampled = np.sort(rng.choice(np.arange(pool_days), size=order_count - 1, replace=pool_days < (order_count - 1)))
            prev_dates = [start + pd.Timedelta(days=int(d)) for d in sampled]
            order_dates = prev_dates + [last_purchase]
        order_dates = sorted(pd.Timestamp(d).floor('D') for d in order_dates)

        net_total = float(max(getattr(row, 'total_spend_365d'), 0.0))
        qty_total = int(max(round(float(getattr(row, 'total_qty_365d'))), order_count))
        if net_total <= 0:
            continue
        amount_weights = rng.dirichlet(np.ones(order_count, dtype=float))
        net_amounts = (amount_weights * net_total).round(2)
        if order_count > 0:
            net_amounts[-1] = round(float(net_total - net_amounts[:-1].sum()), 2)
        quantities = _allocate_integer(qty_total, order_count, rng)

        cat_share = float(np.clip(_safe_ratio([getattr(row, 'spend_in_offer_category_365d')], [max(getattr(row, 'total_spend_365d'), 1e-6)])[0], 0.0, 1.0))
        match_orders = int(np.clip(round(cat_share * order_count), 0, order_count))
        generic_categories = _generic_categories(pd.Series(row._asdict()))
        used_offer_flags = np.array([True] * match_orders + [False] * (order_count - match_orders), dtype=bool)
        rng.shuffle(used_offer_flags)
        coupon_use_prob = float(np.clip(0.08 + 0.35 * cat_share + 0.08 * (float(getattr(row, 'offervalue')) / 3.0), 0.02, 0.70))
        discount_rate = float(np.clip(0.04 + 0.18 * coupon_use_prob, 0.03, 0.22))
        for idx, order_date in enumerate(order_dates):
            minutes = int(rng.integers(9 * 60, 21 * 60))
            ts = order_date + pd.Timedelta(minutes=minutes)
            coupon_used = int(rng.random() < coupon_use_prob)
            net_amount = float(max(net_amounts[idx], 1.0))
            gross_amount = round(net_amount / max(1.0 - discount_rate, 0.70), 2)
            discount_amount = round(max(gross_amount - net_amount, 0.0), 2)
            if bool(used_offer_flags[idx]):
                item_category = _category_label(int(getattr(row, 'category')), 'OFFERCAT')
            else:
                item_category = str(rng.choice(generic_categories))
            rows.append(
                {
                    'order_id': next_order_id,
                    'customer_id': cid,
                    'order_time': ts,
                    'item_category': item_category,
                    'quantity': int(max(int(quantities[idx]), 1)),
                    'gross_amount': gross_amount,
                    'discount_amount': discount_amount,
                    'net_amount': round(net_amount, 2),
                    'coupon_used': coupon_used,
                }
            )
            next_order_id += 1
    return pd.DataFrame(rows)


def _build_exposures(base: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposures_rows: list[dict[str, object]] = []
    contacted_rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        cid = int(getattr(row, 'id'))
        rng = np.random.default_rng(int(seed) * 1_000_033 + cid)
        offerdate = pd.Timestamp(getattr(row, 'offerdate')).floor('D')
        exposure_time = offerdate + pd.Timedelta(hours=9) + pd.Timedelta(minutes=int(rng.integers(0, 180)))
        campaign_type = _campaign_type(float(getattr(row, 'offervalue')))
        coupon_cost = _coupon_cost(float(getattr(row, 'offervalue')))
        exposures_rows.append(
            {
                'exposure_id': f'{cid}_offer_{seed}',
                'customer_id': cid,
                'exposure_time': exposure_time,
                'campaign_type': campaign_type,
                'coupon_cost': coupon_cost,
            }
        )
        contacted_rows.append(
            {
                'customer_id': cid,
                'assigned_at': exposure_time,
                'campaign_type': campaign_type,
                'coupon_cost': coupon_cost,
                'treatment_flag': 1,
                'treatment_group': 'contacted',
                'exposure_count': 1,
                'redeemed_campaigns': int(max(int(getattr(row, 'repeater_flag')), 0)),
                'coupon_redeem_count_total': int(max(int(getattr(row, 'repeattrips')), 0)),
            }
        )
    return pd.DataFrame(exposures_rows), pd.DataFrame(contacted_rows)


def _build_treatment_assignments(customers: pd.DataFrame, contacted: pd.DataFrame, lift: pd.DataFrame) -> pd.DataFrame:
    out = customers[['customer_id', 'signup_date']].merge(contacted, on='customer_id', how='left').merge(lift, on='customer_id', how='left')
    out['assigned_at'] = pd.to_datetime(out['assigned_at']).fillna(pd.to_datetime(out['signup_date']))
    out['campaign_type'] = out['campaign_type'].fillna('TypeA')
    out['coupon_cost'] = pd.to_numeric(out['coupon_cost'], errors='coerce').fillna(0).round().astype(int)
    out['treatment_flag'] = pd.to_numeric(out['treatment_flag'], errors='coerce').fillna(0).astype(int)
    out['treatment_group'] = np.where(out['treatment_flag'].eq(1), 'contacted', 'control')
    out['exposure_count'] = pd.to_numeric(out.get('exposure_count', 0), errors='coerce').fillna(0).astype(int)
    out['redeemed_campaigns'] = pd.to_numeric(out.get('redeemed_campaigns', 0), errors='coerce').fillna(0).astype(int)
    out['coupon_redeem_count_total'] = pd.to_numeric(out.get('coupon_redeem_count_total', 0), errors='coerce').fillna(0).astype(int)
    out['treatment_lift_base'] = pd.to_numeric(out.get('treatment_lift_base', 0.0), errors='coerce').fillna(0.0)
    return out[['customer_id', 'treatment_group', 'treatment_flag', 'campaign_type', 'coupon_cost', 'assigned_at', 'exposure_count', 'redeemed_campaigns', 'coupon_redeem_count_total', 'treatment_lift_base']]


def _build_events(orders: pd.DataFrame, exposures: pd.DataFrame, customers: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=['event_id', 'customer_id', 'timestamp', 'event_type', 'session_id', 'item_category'])
    customer_lookup = customers.set_index('customer_id')[['coupon_affinity', 'price_sensitivity']]
    event_rows: list[dict[str, object]] = []
    event_id = 1
    for row in orders.itertuples(index=False):
        cid = int(getattr(row, 'customer_id'))
        order_time = pd.Timestamp(getattr(row, 'order_time'))
        session_id = f's_{getattr(row, "order_id")}'
        rng = np.random.default_rng(int(seed) * 2_000_003 + int(getattr(row, 'order_id')))
        coupon_affinity = float(customer_lookup.loc[cid, 'coupon_affinity']) if cid in customer_lookup.index else 0.2
        price_sensitivity = float(customer_lookup.loc[cid, 'price_sensitivity']) if cid in customer_lookup.index else 0.2
        base_events = [
            ('visit', order_time - pd.Timedelta(minutes=int(rng.integers(18, 35)))),
            ('page_view', order_time - pd.Timedelta(minutes=int(rng.integers(12, 18)))),
        ]
        if rng.random() < 0.65:
            base_events.append(('search', order_time - pd.Timedelta(minutes=int(rng.integers(7, 12)))))
        if rng.random() < 0.78:
            base_events.append(('add_to_cart', order_time - pd.Timedelta(minutes=int(rng.integers(2, 7)))))
        if rng.random() < (0.04 + 0.10 * price_sensitivity):
            base_events.append(('support_contact', order_time - pd.Timedelta(minutes=int(rng.integers(1, 6)))))
        base_events.append(('purchase', order_time))
        for event_type, ts in sorted(base_events, key=lambda x: x[1]):
            event_rows.append(
                {
                    'event_id': f'evt_{event_id}',
                    'customer_id': cid,
                    'timestamp': ts,
                    'event_type': event_type,
                    'session_id': session_id,
                    'item_category': getattr(row, 'item_category'),
                }
            )
            event_id += 1
    for row in exposures.itertuples(index=False):
        rng = np.random.default_rng(int(seed) * 3_000_001 + int(getattr(row, 'customer_id')))
        open_count = 1 + int(rng.random() < 0.30)
        for open_idx in range(open_count):
            ts = pd.Timestamp(getattr(row, 'exposure_time')) + pd.Timedelta(minutes=int(10 + open_idx * 25 + rng.integers(0, 25)))
            event_rows.append(
                {
                    'event_id': f'evt_{event_id}',
                    'customer_id': int(getattr(row, 'customer_id')),
                    'timestamp': ts,
                    'event_type': 'coupon_open',
                    'session_id': f'camp_{getattr(row, "customer_id")}_{open_idx}',
                    'item_category': str(getattr(row, 'campaign_type')),
                }
            )
            event_id += 1
    return pd.DataFrame(event_rows)



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
    if customers.empty:
        return pd.DataFrame(columns=['customer_id', 'snapshot_date'])

    signup_min = pd.to_datetime(customers['signup_date']).min().floor('D')
    order_max = pd.to_datetime(orders['order_time']).max().floor('D') if len(orders) else signup_min
    exposure_max = pd.to_datetime(exposures['exposure_time']).max().floor('D') if len(exposures) else signup_min
    end_date = max(order_max, exposure_max) + pd.Timedelta(days=112)
    snapshot_dates = pd.date_range(
        start=signup_min + pd.Timedelta(days=snapshot_frequency_days - 1),
        end=end_date,
        freq=f'{int(snapshot_frequency_days)}D',
    )
    snapshot_ord = snapshot_dates.values.astype('datetime64[D]').astype('int64')

    visit_dates = (
        events.loc[events['event_type'].eq('visit')]
        .assign(date=pd.to_datetime(events['timestamp']).dt.floor('D'))
        .groupby('customer_id')['date']
        .apply(lambda s: np.sort(pd.to_datetime(s).values.astype('datetime64[D]').astype('int64')))
        .to_dict()
    )
    purchase_dates = (
        orders.assign(date=pd.to_datetime(orders['order_time']).dt.floor('D'))
        .groupby('customer_id')['date']
        .apply(lambda s: np.sort(pd.to_datetime(s).values.astype('datetime64[D]').astype('int64')))
        .to_dict()
    )
    spend_pairs = {}
    if len(orders):
        tmp = orders.assign(date=pd.to_datetime(orders['order_time']).dt.floor('D'))
        for cid, grp in tmp.groupby('customer_id'):
            ords = pd.to_datetime(grp['date']).values.astype('datetime64[D]').astype('int64')
            amts = pd.to_numeric(grp['net_amount'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
            order = np.argsort(ords, kind='stable')
            spend_pairs[int(cid)] = (ords[order], np.cumsum(amts[order], dtype=float))
    exposure_dates = (
        exposures.assign(date=pd.to_datetime(exposures['exposure_time']).dt.floor('D'))
        .groupby('customer_id')['date']
        .apply(lambda s: np.sort(pd.to_datetime(s).values.astype('datetime64[D]').astype('int64')))
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    for customer_id, signup_date in customers[['customer_id', 'signup_date']].itertuples(index=False):
        cid = int(customer_id)
        signup_ord = pd.Timestamp(signup_date).floor('D').to_datetime64().astype('datetime64[D]').astype('int64')
        visits = visit_dates.get(cid, np.empty(0, dtype='int64'))
        purchases = purchase_dates.get(cid, np.empty(0, dtype='int64'))
        spend_ords, spend_cumsum = spend_pairs.get(cid, (np.empty(0, dtype='int64'), np.empty(0, dtype='float64')))
        exposures_c = exposure_dates.get(cid, np.empty(0, dtype='int64'))

        start_idx = int(np.searchsorted(snapshot_ord, signup_ord, side='left'))
        for snap in snapshot_ord[start_idx:]:
            visit_count = int(np.searchsorted(visits, snap, side='right'))
            purchase_count = int(np.searchsorted(purchases, snap, side='right'))
            exposure_count = int(np.searchsorted(exposures_c, snap, side='right'))
            spend_count = int(np.searchsorted(spend_ords, snap, side='right'))

            last_visit_ord = visits[visit_count - 1] if visit_count else None
            last_purchase_ord = purchases[purchase_count - 1] if purchase_count else None
            anchor_ord = last_purchase_ord if last_purchase_ord is not None else last_visit_ord
            inactivity_days = int(snap - anchor_ord) if anchor_ord is not None else int(snap - signup_ord)

            if inactivity_days >= churn_inactivity_days:
                status = 'churn_risk'
            elif inactivity_days >= dormant_inactivity_days:
                status = 'dormant'
            else:
                status = 'active'

            visit_28 = int(visit_count - np.searchsorted(visits, snap - 28, side='left')) if visit_count else 0
            purchase_28 = int(purchase_count - np.searchsorted(purchases, snap - 28, side='left')) if purchase_count else 0
            exposure_28 = int(exposure_count - np.searchsorted(exposures_c, snap - 28, side='left')) if exposure_count else 0
            monetary_total = float(spend_cumsum[spend_count - 1]) if spend_count else 0.0

            rows.append(
                {
                    'customer_id': cid,
                    'snapshot_date': pd.Timestamp(np.datetime64(int(snap), 'D')),
                    'last_visit_date': pd.Timestamp(np.datetime64(int(last_visit_ord), 'D')) if last_visit_ord is not None else pd.NaT,
                    'last_purchase_date': pd.Timestamp(np.datetime64(int(last_purchase_ord), 'D')) if last_purchase_ord is not None else pd.NaT,
                    'visits_total': visit_count,
                    'purchases_total': purchase_count,
                    'monetary_total': round(monetary_total, 2),
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


def _export_seed(seed_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(seed_dir / f'{name}.csv', index=False)


def _build_tables_for_seed(base: pd.DataFrame, *, seed: int, snapshot_frequency_days: int, dormant_inactivity_days: int, churn_inactivity_days: int) -> dict[str, pd.DataFrame]:
    customers, lift = _build_customers(base)
    exposures, contacted = _build_exposures(base, seed=seed)
    treatment = _build_treatment_assignments(customers, contacted, lift)
    orders = _build_orders(base, seed=seed)
    events = _build_events(orders, exposures, customers, seed=seed)
    snapshots = _build_state_snapshots(
        customers,
        events,
        orders,
        exposures,
        snapshot_frequency_days=snapshot_frequency_days,
        dormant_inactivity_days=dormant_inactivity_days,
        churn_inactivity_days=churn_inactivity_days,
    )
    customer_summary = _build_customer_summary(customers, orders, exposures, snapshots)
    cohort_retention = _build_cohort_retention(customers, orders)
    return {
        'customers': customers.sort_values('customer_id').reset_index(drop=True),
        'events': events.sort_values(['customer_id', 'timestamp']).reset_index(drop=True),
        'orders': orders.sort_values(['customer_id', 'order_time']).reset_index(drop=True),
        'state_snapshots': snapshots.sort_values(['customer_id', 'snapshot_date']).reset_index(drop=True),
        'campaign_exposures': exposures.sort_values(['customer_id', 'exposure_time']).reset_index(drop=True),
        'treatment_assignments': treatment.sort_values('customer_id').reset_index(drop=True),
        'customer_summary': customer_summary.sort_values('customer_id').reset_index(drop=True),
        'cohort_retention': cohort_retention,
    }


def import_acquire_valued_shoppers(config: ImportConfig) -> dict[str, object]:
    artifact_root = _artifact_root(config)
    work_dir = artifact_root / 'external_imports' / 'acquire_valued_shoppers'
    raw_grid_root = artifact_root / 'raw_grid'
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_grid_root.mkdir(parents=True, exist_ok=True)

    base = _read_inputs(config)
    manifest_seeds: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = raw_grid_root / f'seed_{int(seed)}'
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        tables = _build_tables_for_seed(
            base,
            seed=int(seed),
            snapshot_frequency_days=int(config.snapshot_frequency_days),
            dormant_inactivity_days=int(config.dormant_inactivity_days),
            churn_inactivity_days=int(config.churn_inactivity_days),
        )
        _export_seed(seed_dir, tables)
        manifest_seeds.append(
            {
                'seed': int(seed),
                'seed_dir': str(seed_dir),
                'customer_count': int(tables['customers']['customer_id'].nunique()),
                'order_count': int(len(tables['orders'])),
                'event_count': int(len(tables['events'])),
                'snapshot_count': int(len(tables['state_snapshots'])),
                'date_min': str(pd.to_datetime(tables['orders']['order_time']).min().date()) if len(tables['orders']) else None,
                'date_max': str(pd.to_datetime(tables['orders']['order_time']).max().date()) if len(tables['orders']) else None,
            }
        )

    manifest = {
        'aggregate_path': str(config.aggregate_path),
        'train_history_path': str(config.train_history_path),
        'offers_path': str(config.offers_path),
        'artifacts_root': str(artifact_root),
        'seeds': [int(s) for s in config.seeds],
        'household_limit': None if config.household_limit is None else int(config.household_limit),
        'customer_count': int(len(base)),
        'offer_date_min': str(pd.to_datetime(base['offerdate']).min().date()),
        'offer_date_max': str(pd.to_datetime(base['offerdate']).max().date()),
        'seed_manifests': manifest_seeds,
    }
    (work_dir / 'import_manifest.json').write_text(pd.Series(manifest, dtype='object').to_json(force_ascii=False, indent=2), encoding='utf-8')
    return manifest
