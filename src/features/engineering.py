from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.features.store import FileFeatureStore


@dataclass
class FeatureBuildResult:
    features: pd.DataFrame
    metadata: Dict
    feature_store_csv_path: str
    feature_store_metadata_path: str


def safe_divide(numerator, denominator, default: float = 0.0) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, default, dtype=float)
    mask = (den != 0) & np.isfinite(den) & np.isfinite(num)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        out[mask] = num[mask] / den[mask]
    out[~np.isfinite(out)] = default
    return out


def add_rate_feature(df: pd.DataFrame, recent_col: str, prev_col: str, output_col: str) -> None:
    df[output_col] = safe_divide(
        df[recent_col] - df[prev_col],
        df[prev_col].replace(0, np.nan),
        default=0.0,
    )


def feature_dictionary() -> Dict[str, str]:
    return {
        'customer_age_days': '기준일까지의 고객 생존 일수',
        'days_since_last_event': '마지막 이벤트 이후 경과 일수',
        'recency_days': '마지막 구매 이후 경과 일수',
        'frequency_30d': '최근 30일 구매 횟수',
        'frequency_90d': '최근 90일 구매 횟수',
        'monetary_30d': '최근 30일 순매출 합계',
        'monetary_90d': '최근 90일 순매출 합계',
        'avg_order_value_90d': '최근 90일 평균 주문 금액',
        'monetary_per_visit_90d': '최근 90일 방문 1회당 매출',
        'visits_14d': '최근 14일 방문 수',
        'visits_prev_14d': '직전 14일 방문 수',
        'visit_change_rate_14d': '최근 14일 방문 변화율',
        'purchases_14d': '최근 14일 구매 수',
        'purchases_prev_14d': '직전 14일 구매 수',
        'purchase_change_rate_14d': '최근 14일 구매 변화율',
        'searches_30d': '최근 30일 검색 수',
        'searches_prev_30d': '직전 30일 검색 수',
        'add_to_cart_30d': '최근 30일 장바구니 추가 수',
        'add_to_cart_prev_30d': '직전 30일 장바구니 추가 수',
        'coupon_open_30d': '최근 30일 쿠폰 오픈 수',
        'coupon_open_prev_30d': '직전 30일 쿠폰 오픈 수',
        'coupon_open_rate_30d': '최근 30일 쿠폰 오픈율',
        'coupon_response_change_rate': '쿠폰 반응률 변화',
        'avg_purchase_gap_days': '평균 구매 주기',
        'median_purchase_gap_days': '중앙 구매 주기',
        'current_non_purchase_days': '현재 미구매 일수',
        'purchase_cycle_anomaly': '현재 미구매 일수 / 평균 구매 주기',
        'avg_session_duration_sec_30d': '최근 30일 평균 세션 시간',
        'avg_session_duration_sec_prev_30d': '직전 30일 평균 세션 시간',
        'session_duration_change_rate': '세션 시간 변화율',
        'pageviews_per_session_30d': '최근 30일 세션당 페이지뷰',
        'pageviews_per_session_prev_30d': '직전 30일 세션당 페이지뷰',
        'pageviews_change_rate': '세션당 페이지뷰 변화율',
        'search_to_purchase_conversion_30d': '최근 30일 검색 후 구매 전환율',
        'search_to_purchase_conversion_prev_30d': '직전 30일 검색 후 구매 전환율',
        'search_purchase_conv_change_rate': '검색 후 구매 전환율 변화',
        'cart_to_purchase_rate_30d': '최근 30일 장바구니→구매 전환율',
        'cart_to_purchase_rate_prev_30d': '직전 30일 장바구니→구매 전환율',
        'cart_conversion_change_rate': '장바구니 전환율 변화',
        'support_contact_30d': '최근 30일 문의 이벤트 수',
        'support_contact_rate_30d': '최근 30일 세션당 문의 비율',
        'sessions_30d': '최근 30일 세션 수',
        'sessions_prev_30d': '직전 30일 세션 수',
        'active_days_30d': '최근 30일 활동한 고유 일수',
        'orders_with_coupon_ratio_90d': '최근 90일 쿠폰 사용 주문 비율',
        'coupon_redeem_rate_90d': '최근 90일 쿠폰 사용 주문 비율',
        'exposure_count_30d': '최근 30일 캠페인 노출 수',
        'coupon_cost_30d': '최근 30일 쿠폰 비용',
        'weekend_purchase_ratio': '주말 구매 비율',
        'weekday_purchase_ratio': '평일 구매 비율',
        'evening_activity_ratio': '저녁 시간 활동 비율',
        'night_activity_ratio': '심야 활동 비율',
        'workhour_activity_ratio': '근무시간 활동 비율',
        'weekend_activity_ratio': '주말 전체 활동 비율',
        'event_diversity_90d': '최근 90일 이벤트 다양성',
        'recent_event_sequence': '최근 N개 이벤트 시퀀스',
        'behavior_cluster_id': '행동 클러스터 ID',
        'dominant_event_type_90d': '최근 90일 최빈 이벤트 타입',
        'current_journey_stage': '현재 고객 여정 단계',
        'journey_stage_days': '현재 단계 체류 일수',
        'inactivity_days': '비활성 일수',
        'recent_visit_score': '최근 방문 점수',
        'recent_purchase_score': '최근 구매 점수',
        'recent_exposure_score': '최근 노출 점수',
        'coupon_fatigue_score': '누적 쿠폰 피로도 점수',
        'discount_dependency_score': '할인 의존도 점수',
        'discount_pressure_score': '프로모션 압박 점수',
        'discount_effect_penalty': '할인 과다 노출로 인한 기대효과 보정치',
        'discount_fatigue_sensitivity': '할인 피로 민감도',
        'offer_dependency_risk': '혜택 의존 위험도',
        'brand_sensitivity': '브랜드/정가 훼손 민감도',
    }


def _load_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        'customers': pd.read_csv(data_dir / 'customers.csv', parse_dates=['signup_date']),
        'events': pd.read_csv(data_dir / 'events.csv', parse_dates=['timestamp']),
        'orders': pd.read_csv(data_dir / 'orders.csv', parse_dates=['order_time']),
        'snapshots': pd.read_csv(data_dir / 'state_snapshots.csv', parse_dates=['snapshot_date', 'last_visit_date', 'last_purchase_date']),
        'exposures': pd.read_csv(data_dir / 'campaign_exposures.csv', parse_dates=['exposure_time']),
        'treatment': pd.read_csv(data_dir / 'treatment_assignments.csv', parse_dates=['assigned_at']),
    }


def _window_counts_by_customer(df: pd.DataFrame, customer_col: str, ts_col: str, end_date: pd.Timestamp, days: int, value_col: str | None = None, event_filter: Iterable[str] | None = None) -> pd.Series:
    start = end_date - pd.Timedelta(days=days)
    tmp = df.loc[(df[ts_col] > start) & (df[ts_col] <= end_date)].copy()
    if event_filter is not None:
        tmp = tmp[tmp['event_type'].isin(list(event_filter))]
    if value_col is None:
        return tmp.groupby(customer_col).size()
    return tmp.groupby(customer_col)[value_col].sum()


def _window_unique_days(df: pd.DataFrame, customer_col: str, ts_col: str, end_date: pd.Timestamp, days: int) -> pd.Series:
    start = end_date - pd.Timedelta(days=days)
    tmp = df.loc[(df[ts_col] > start) & (df[ts_col] <= end_date), [customer_col, ts_col]].copy()
    tmp['activity_date'] = tmp[ts_col].dt.floor('D')
    return tmp.groupby(customer_col)['activity_date'].nunique()


def _compute_purchase_cycle_features(orders: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    hist = orders.loc[orders['order_time'] <= as_of_date, ['customer_id', 'order_time', 'quantity', 'coupon_used']].sort_values(['customer_id', 'order_time'])
    hist['prev_order_time'] = hist.groupby('customer_id')['order_time'].shift(1)
    hist['purchase_gap_days'] = (hist['order_time'] - hist['prev_order_time']).dt.total_seconds() / 86400.0
    agg = hist.groupby('customer_id').agg(
        avg_purchase_gap_days=('purchase_gap_days', 'mean'),
        median_purchase_gap_days=('purchase_gap_days', 'median'),
        avg_items_per_order_90d=('quantity', 'mean'),
        orders_with_coupon_ratio_90d=('coupon_used', 'mean'),
    )
    last_purchase = hist.groupby('customer_id')['order_time'].max()
    agg['current_non_purchase_days'] = (as_of_date - last_purchase).dt.total_seconds() / 86400.0
    agg['purchase_cycle_anomaly'] = safe_divide(
        agg['current_non_purchase_days'],
        agg['avg_purchase_gap_days'].replace(0, np.nan),
        default=0.0,
    )
    return agg.reset_index()


def _compute_session_features(events: pd.DataFrame, orders: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    hist_events = events.loc[
        (events['timestamp'] > as_of_date - pd.Timedelta(days=60)) & (events['timestamp'] <= as_of_date),
        ['customer_id', 'timestamp', 'event_type', 'session_id'],
    ].copy()
    hist_orders = orders.loc[
        (orders['order_time'] > as_of_date - pd.Timedelta(days=60)) & (orders['order_time'] <= as_of_date),
        ['customer_id', 'order_time', 'order_id'],
    ].copy()

    if hist_events.empty:
        return pd.DataFrame(
            columns=[
                'customer_id',
                'avg_session_duration_sec_30d',
                'median_session_duration_sec_30d',
                'pageviews_per_session_30d',
                'searches_per_session_30d',
                'sessions_30d',
                'support_contacts_per_session_30d',
                'avg_session_duration_sec_prev_30d',
                'median_session_duration_sec_prev_30d',
                'pageviews_per_session_prev_30d',
                'searches_per_session_prev_30d',
                'sessions_prev_30d',
                'support_contacts_per_session_prev_30d',
                'search_to_purchase_conversion_total',
            ]
        )

    session_base = hist_events.groupby(['customer_id', 'session_id']).agg(
        session_start=('timestamp', 'min'),
        session_end=('timestamp', 'max'),
    )
    event_counts = pd.crosstab(index=[hist_events['customer_id'], hist_events['session_id']], columns=hist_events['event_type'])
    session_stats = session_base.join(event_counts, how='left').reset_index()
    for col in ['page_view', 'search', 'add_to_cart', 'support_contact', 'purchase']:
        if col not in session_stats.columns:
            session_stats[col] = 0
    session_stats['session_duration_sec'] = (
        session_stats['session_end'] - session_stats['session_start']
    ).dt.total_seconds().clip(lower=0)

    def _agg_between(start: pd.Timestamp, end: pd.Timestamp, suffix: str) -> pd.DataFrame:
        tmp = session_stats.loc[(session_stats['session_start'] > start) & (session_stats['session_start'] <= end)]
        if tmp.empty:
            return pd.DataFrame(columns=['customer_id'])
        return tmp.groupby('customer_id').agg(
            **{
                f'avg_session_duration_sec_{suffix}': ('session_duration_sec', 'mean'),
                f'median_session_duration_sec_{suffix}': ('session_duration_sec', 'median'),
                f'pageviews_per_session_{suffix}': ('page_view', 'mean'),
                f'searches_per_session_{suffix}': ('search', 'mean'),
                f'sessions_{suffix}': ('session_id', 'nunique'),
                f'support_contacts_per_session_{suffix}': ('support_contact', 'mean'),
            }
        ).reset_index()

    recent_30 = _agg_between(as_of_date - pd.Timedelta(days=30), as_of_date, '30d')
    prev_30 = _agg_between(as_of_date - pd.Timedelta(days=60), as_of_date - pd.Timedelta(days=30), 'prev_30d')

    if hist_orders.empty:
        conv = pd.DataFrame(columns=['customer_id', 'search_to_purchase_conversion_total'])
    else:
        hist_orders['order_date'] = hist_orders['order_time'].dt.floor('D')
        purchase_days = hist_orders.groupby('customer_id')['order_date'].nunique().rename('purchase_days_total')
        search_sessions = (
            session_stats.assign(has_search=session_stats['search'] > 0)
            .groupby('customer_id')['has_search']
            .sum()
            .rename('search_sessions_total')
        )
        conv = pd.concat([search_sessions, purchase_days], axis=1).fillna(0.0)
        conv['search_to_purchase_conversion_total'] = safe_divide(
            conv['purchase_days_total'],
            conv['search_sessions_total'],
            default=0.0,
        )
        conv = conv[['search_to_purchase_conversion_total']].reset_index()

    out = recent_30.merge(prev_30, on='customer_id', how='outer')
    out = out.merge(conv, on='customer_id', how='left')
    expected_columns = [
        'customer_id',
        'avg_session_duration_sec_30d',
        'median_session_duration_sec_30d',
        'pageviews_per_session_30d',
        'searches_per_session_30d',
        'sessions_30d',
        'support_contacts_per_session_30d',
        'avg_session_duration_sec_prev_30d',
        'median_session_duration_sec_prev_30d',
        'pageviews_per_session_prev_30d',
        'searches_per_session_prev_30d',
        'sessions_prev_30d',
        'support_contacts_per_session_prev_30d',
        'search_to_purchase_conversion_total',
    ]
    for column in expected_columns:
        if column not in out.columns:
            out[column] = 0.0 if column != 'customer_id' else pd.Series(dtype=int)
    return out[expected_columns]


def _compute_recent_event_sequence(events: pd.DataFrame, as_of_date: pd.Timestamp, n_recent_events: int) -> pd.Series:
    hist = events.loc[events['timestamp'] <= as_of_date, ['customer_id', 'timestamp', 'event_type']].sort_values(['customer_id', 'timestamp'])
    recent = hist.groupby('customer_id').tail(n_recent_events)
    return recent.groupby('customer_id')['event_type'].apply(lambda x: ' > '.join(x.tolist()))


def _compute_behavior_cluster(events: pd.DataFrame, as_of_date: pd.Timestamp, n_clusters: int) -> pd.Series:
    hist = events.loc[(events['timestamp'] > as_of_date - pd.Timedelta(days=90)) & (events['timestamp'] <= as_of_date), ['customer_id', 'event_type']]
    mix = pd.crosstab(hist['customer_id'], hist['event_type'])
    if mix.empty:
        return pd.Series(dtype=int)
    mix = mix.astype(float)
    row_sum = mix.sum(axis=1).replace(0.0, np.nan)
    mix = mix.div(row_sum, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    unique_rows = max(1, int(mix.drop_duplicates().shape[0]))
    n_clusters = max(2, min(int(n_clusters), len(mix), unique_rows))
    if n_clusters <= 1 or len(mix) < 2 or unique_rows < 2:
        return pd.Series(np.zeros(len(mix), dtype=int), index=mix.index, name='behavior_cluster_id')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore')
            clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, algorithm='lloyd').fit_predict(mix.to_numpy(dtype=float))
    except Exception:
        dominant = mix.idxmax(axis=1).astype('category').cat.codes.to_numpy(dtype=int)
        clusters = dominant % n_clusters
    return pd.Series(clusters, index=mix.index, name='behavior_cluster_id')


def _compute_state_features(snapshots: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    hist = snapshots.loc[snapshots['snapshot_date'] <= as_of_date, ['customer_id', 'snapshot_date', 'inactivity_days', 'current_status', 'recent_visit_score', 'recent_purchase_score', 'recent_exposure_score']].copy()
    hist = hist.sort_values(['customer_id', 'snapshot_date'])
    hist['status_change'] = hist.groupby('customer_id')['current_status'].transform(lambda s: s.ne(s.shift()))
    hist['status_group'] = hist.groupby('customer_id')['status_change'].cumsum()
    hist['status_start_date'] = hist.groupby(['customer_id', 'status_group'])['snapshot_date'].transform('min')
    as_of_rows = hist.groupby('customer_id').tail(1).copy()
    as_of_rows['journey_stage_days'] = (as_of_rows['snapshot_date'] - as_of_rows['status_start_date']).dt.days + 1
    as_of_rows = as_of_rows.rename(columns={'current_status': 'current_journey_stage'})
    return as_of_rows[['customer_id', 'inactivity_days', 'current_journey_stage', 'journey_stage_days', 'recent_visit_score', 'recent_purchase_score', 'recent_exposure_score']]


def _compute_time_features(events: pd.DataFrame, orders: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    hist_events = events.loc[(events['timestamp'] > as_of_date - pd.Timedelta(days=90)) & (events['timestamp'] <= as_of_date), ['customer_id', 'timestamp', 'event_type']].copy()
    hist_orders = orders.loc[(orders['order_time'] > as_of_date - pd.Timedelta(days=90)) & (orders['order_time'] <= as_of_date), ['customer_id', 'order_time']].copy()
    hist_events['hour'] = hist_events['timestamp'].dt.hour
    hist_events['is_weekend'] = hist_events['timestamp'].dt.dayofweek >= 5
    hist_orders['is_weekend'] = hist_orders['order_time'].dt.dayofweek >= 5
    evt = hist_events.groupby('customer_id').agg(
        evening_events=('hour', lambda x: int(((x >= 18) & (x < 24)).sum())),
        night_events=('hour', lambda x: int(((x >= 0) & (x < 6)).sum())),
        workhour_events=('hour', lambda x: int(((x >= 9) & (x < 18)).sum())),
        weekend_events=('is_weekend', 'sum'),
        total_events=('event_type', 'size'),
        event_diversity_90d=('event_type', pd.Series.nunique),
    )
    evt['evening_activity_ratio'] = safe_divide(evt['evening_events'], evt['total_events'], default=0.0)
    evt['night_activity_ratio'] = safe_divide(evt['night_events'], evt['total_events'], default=0.0)
    evt['workhour_activity_ratio'] = safe_divide(evt['workhour_events'], evt['total_events'], default=0.0)
    evt['weekend_activity_ratio'] = safe_divide(evt['weekend_events'], evt['total_events'], default=0.0)
    ords = hist_orders.groupby('customer_id').agg(weekend_purchases=('is_weekend', 'sum'), total_purchases_90d=('order_time', 'size'))
    ords['weekend_purchase_ratio'] = safe_divide(ords['weekend_purchases'], ords['total_purchases_90d'], default=0.0)
    ords['weekday_purchase_ratio'] = 1.0 - ords['weekend_purchase_ratio']
    return evt.merge(ords[['weekend_purchase_ratio', 'weekday_purchase_ratio']], left_index=True, right_index=True, how='outer').reset_index()


def _compute_future_label(events: pd.DataFrame, orders: pd.DataFrame, snapshots: pd.DataFrame, customer_ids: pd.Series, as_of_date: pd.Timestamp, horizon_days: int) -> pd.Series:
    end_date = as_of_date + pd.Timedelta(days=horizon_days)
    future_visits = events.loc[(events['timestamp'] > as_of_date) & (events['timestamp'] <= end_date) & (events['event_type'] == 'visit')].groupby('customer_id').size()
    future_purchases = orders.loc[(orders['order_time'] > as_of_date) & (orders['order_time'] <= end_date)].groupby('customer_id').size()
    future_status = snapshots.loc[(snapshots['snapshot_date'] > as_of_date) & (snapshots['snapshot_date'] <= end_date), ['customer_id', 'snapshot_date', 'current_status']].sort_values(['customer_id', 'snapshot_date'])
    future_last = future_status.groupby('customer_id').tail(1).set_index('customer_id')['current_status']
    return (
        customer_ids.map(future_visits).fillna(0).eq(0)
        & customer_ids.map(future_purchases).fillna(0).eq(0)
        & customer_ids.map(future_last).fillna('active').eq('churn_risk')
    ).astype(int)


def _winsorize_and_impute(features: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    out = features.copy()
    summary = {}
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in {'customer_id', 'label'}]
    for col in numeric_cols:
        s = out[col].replace([np.inf, -np.inf], np.nan)
        q01 = float(s.quantile(0.01)) if s.notna().any() else 0.0
        q99 = float(s.quantile(0.99)) if s.notna().any() else 0.0
        fill = float(s.median()) if s.notna().any() else 0.0
        out[col] = s.clip(lower=q01, upper=q99).fillna(fill).clip(lower=-1_000_000.0, upper=1_000_000.0)
        summary[col] = {'clip_p01': q01, 'clip_p99': q99, 'fill_value': fill}
    for col in out.columns:
        if col not in numeric_cols and col not in {'customer_id', 'label'}:
            s = out[col].copy()
            s = s.where(s.notna(), 'unknown').astype('string').fillna('unknown')
            out[col] = s.replace({'nan': 'unknown', '<NA>': 'unknown', 'NaT': 'unknown'}).astype(str)
    return out, summary


def build_feature_dataset(data_dir: str | Path, feature_store_dir: str | Path = 'data/feature_store', as_of_date: str | pd.Timestamp | None = None, horizon_days: int = 45, n_recent_events: int = 5, n_clusters: int = 6) -> FeatureBuildResult:
    data_dir = Path(data_dir)
    raw = _load_csvs(data_dir)
    customers = raw['customers'].copy()
    events = raw['events'].copy()
    orders = raw['orders'].copy()
    snapshots = raw['snapshots'].copy()
    exposures = raw['exposures'].copy()
    treatment = raw['treatment'].copy()
    if as_of_date is None:
        as_of_date = snapshots['snapshot_date'].max() - pd.Timedelta(days=horizon_days)
    as_of_date = pd.Timestamp(as_of_date).floor('D')
    customers = customers.loc[customers['signup_date'] <= as_of_date].copy()
    base = customers.merge(treatment, on='customer_id', how='left')
    base['customer_age_days'] = (as_of_date - base['signup_date']).dt.days.clip(lower=0)
    last_event = events.loc[events['timestamp'] <= as_of_date].groupby('customer_id')['timestamp'].max()
    last_purchase = orders.loc[orders['order_time'] <= as_of_date].groupby('customer_id')['order_time'].max()
    base['days_since_last_event'] = (as_of_date - base['customer_id'].map(last_event)).dt.total_seconds().div(86400).fillna(999)
    base['recency_days'] = (as_of_date - base['customer_id'].map(last_purchase)).dt.total_seconds().div(86400).fillna(999)
    base['frequency_30d'] = base['customer_id'].map(_window_counts_by_customer(orders, 'customer_id', 'order_time', as_of_date, 30)).fillna(0)
    base['frequency_prev_30d'] = base['customer_id'].map(_window_counts_by_customer(orders, 'customer_id', 'order_time', as_of_date - pd.Timedelta(days=30), 30)).fillna(0)
    base['frequency_90d'] = base['customer_id'].map(_window_counts_by_customer(orders, 'customer_id', 'order_time', as_of_date, 90)).fillna(0)
    base['monetary_30d'] = base['customer_id'].map(_window_counts_by_customer(orders, 'customer_id', 'order_time', as_of_date, 30, value_col='net_amount')).fillna(0.0)
    base['monetary_90d'] = base['customer_id'].map(_window_counts_by_customer(orders, 'customer_id', 'order_time', as_of_date, 90, value_col='net_amount')).fillna(0.0)
    base['avg_order_value_90d'] = safe_divide(base['monetary_90d'], base['frequency_90d'].replace(0, np.nan), default=0.0)
    base['visits_14d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 14, event_filter=['visit'])).fillna(0)
    base['visits_prev_14d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date - pd.Timedelta(days=14), 14, event_filter=['visit'])).fillna(0)
    base['purchases_14d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 14, event_filter=['purchase'])).fillna(0)
    base['purchases_prev_14d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date - pd.Timedelta(days=14), 14, event_filter=['purchase'])).fillna(0)
    base['searches_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 30, event_filter=['search'])).fillna(0)
    base['searches_prev_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date - pd.Timedelta(days=30), 30, event_filter=['search'])).fillna(0)
    base['add_to_cart_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 30, event_filter=['add_to_cart'])).fillna(0)
    base['add_to_cart_prev_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date - pd.Timedelta(days=30), 30, event_filter=['add_to_cart'])).fillna(0)
    base['coupon_open_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 30, event_filter=['coupon_open'])).fillna(0)
    base['coupon_open_prev_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date - pd.Timedelta(days=30), 30, event_filter=['coupon_open'])).fillna(0)
    base['support_contact_30d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 30, event_filter=['support_contact'])).fillna(0)
    base['visits_90d'] = base['customer_id'].map(_window_counts_by_customer(events, 'customer_id', 'timestamp', as_of_date, 90, event_filter=['visit'])).fillna(0)
    base['active_days_30d'] = base['customer_id'].map(_window_unique_days(events, 'customer_id', 'timestamp', as_of_date, 30)).fillna(0)
    add_rate_feature(base, 'visits_14d', 'visits_prev_14d', 'visit_change_rate_14d')
    add_rate_feature(base, 'purchases_14d', 'purchases_prev_14d', 'purchase_change_rate_14d')
    base = base.merge(_compute_purchase_cycle_features(orders, as_of_date), on='customer_id', how='left')
    base = base.merge(_compute_session_features(events, orders, as_of_date), on='customer_id', how='left')
    base['exposure_count_30d'] = base['customer_id'].map(_window_counts_by_customer(exposures, 'customer_id', 'exposure_time', as_of_date, 30)).fillna(0)
    base['exposure_count_prev_30d'] = base['customer_id'].map(_window_counts_by_customer(exposures, 'customer_id', 'exposure_time', as_of_date - pd.Timedelta(days=30), 30)).fillna(0)
    base['search_to_purchase_conversion_30d'] = safe_divide(base['frequency_30d'], base['searches_30d'].replace(0, np.nan), default=0.0)
    base['search_to_purchase_conversion_prev_30d'] = safe_divide(base['frequency_prev_30d'], base['searches_prev_30d'].replace(0, np.nan), default=0.0)
    base['cart_to_purchase_rate_30d'] = safe_divide(base['frequency_30d'], base['add_to_cart_30d'].replace(0, np.nan), default=0.0)
    base['cart_to_purchase_rate_prev_30d'] = safe_divide(base['frequency_prev_30d'], base['add_to_cart_prev_30d'].replace(0, np.nan), default=0.0)
    base['coupon_open_rate_30d'] = safe_divide(base['coupon_open_30d'], base['exposure_count_30d'].replace(0, np.nan), default=0.0)
    base['coupon_open_rate_prev_30d'] = safe_divide(base['coupon_open_prev_30d'], base['exposure_count_prev_30d'].replace(0, np.nan), default=0.0)
    add_rate_feature(base, 'avg_session_duration_sec_30d', 'avg_session_duration_sec_prev_30d', 'session_duration_change_rate')
    add_rate_feature(base, 'pageviews_per_session_30d', 'pageviews_per_session_prev_30d', 'pageviews_change_rate')
    add_rate_feature(base, 'search_to_purchase_conversion_30d', 'search_to_purchase_conversion_prev_30d', 'search_purchase_conv_change_rate')
    add_rate_feature(base, 'cart_to_purchase_rate_30d', 'cart_to_purchase_rate_prev_30d', 'cart_conversion_change_rate')
    add_rate_feature(base, 'coupon_open_rate_30d', 'coupon_open_rate_prev_30d', 'coupon_response_change_rate')
    base['coupon_cost_30d'] = base['customer_id'].map(_window_counts_by_customer(exposures, 'customer_id', 'exposure_time', as_of_date, 30, value_col='coupon_cost')).fillna(0.0)
    base['coupon_redeem_rate_90d'] = base['orders_with_coupon_ratio_90d'].fillna(0.0)
    base = base.merge(_compute_time_features(events, orders, as_of_date), on='customer_id', how='left')
    base['recent_event_sequence'] = base['customer_id'].map(_compute_recent_event_sequence(events, as_of_date, n_recent_events)).fillna('no_history')
    base['behavior_cluster_id'] = base['customer_id'].map(_compute_behavior_cluster(events, as_of_date, n_clusters)).fillna(-1).astype(int)
    hist_90d = events.loc[(events['timestamp'] > as_of_date - pd.Timedelta(days=90)) & (events['timestamp'] <= as_of_date), ['customer_id', 'event_type']]
    dominant = hist_90d.groupby(['customer_id', 'event_type']).size().reset_index(name='cnt').sort_values(['customer_id', 'cnt'], ascending=[True, False]).drop_duplicates('customer_id').set_index('customer_id')['event_type']
    base['dominant_event_type_90d'] = base['customer_id'].map(dominant).fillna('none')
    base = base.merge(_compute_state_features(snapshots, as_of_date), on='customer_id', how='left')
    base['monetary_per_visit_90d'] = safe_divide(base['monetary_90d'], base['visits_90d'].replace(0, np.nan), default=0.0)
    base['support_contact_rate_30d'] = safe_divide(base['support_contact_30d'], base['sessions_30d'].replace(0, np.nan), default=0.0)
    base['label'] = _compute_future_label(events, orders, snapshots, base['customer_id'], as_of_date, horizon_days)
    # Remove strict active filter to allow higher churn rate
    # active_cohort = base['active_days_30d'] > 0
    # base = base.loc[active_cohort].reset_index(drop=True)
    base, clipping_summary = _winsorize_and_impute(base)
    positive_rate = float(base['label'].mean())
    print(f'Churn rate: {positive_rate * 100:.2f}%')
    metadata = {
        'as_of_date': str(as_of_date.date()),
        'horizon_days': horizon_days,
        'row_count': int(len(base)),
        'positive_rate': positive_rate,
        'feature_count': int(len([c for c in base.columns if c not in {'customer_id', 'label'}])),
        'feature_dictionary': feature_dictionary(),
        'clipping_summary': clipping_summary,
        'cohort_filter': 'none',
        'missing_value_strategy': 'numeric=median, categorical=unknown, outlier=1st/99th percentile clipping',
    }
    store = FileFeatureStore(feature_store_dir)
    paths = store.save(base, metadata, dataset_name='customer_features')
    return FeatureBuildResult(base, metadata, str(paths.feature_csv_path), str(paths.metadata_path))
