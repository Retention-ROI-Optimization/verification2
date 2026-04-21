from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .event_rules import (
    compute_add_to_cart_probability,
    compute_browse_probability,
    compute_coupon_open_probability,
    compute_coupon_redeem_probability,
    compute_purchase_probability,
    compute_remove_cart_probability,
    compute_search_probability,
    compute_visit_probability,
)
from .order_builder import build_orders
from .state_tracker import StateTracker


_ITEM_CATEGORIES = np.array(["fashion", "beauty", "personal_care", "grocery", "sports", "health"], dtype=object)
_CATEGORY_PROBS = np.array([0.20, 0.18, 0.18, 0.16, 0.14, 0.14], dtype=float)


def _empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "customer_id",
            "timestamp",
            "event_type",
            "session_id",
            "item_category",
            "quantity",
        ]
    )


def _sample_session_start_minutes(device_types: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(device_types)
    if n == 0:
        return np.array([], dtype=int)

    hours = np.zeros(n, dtype=int)
    mobile_mask = device_types == "mobile"
    desktop_mask = device_types == "desktop"
    tablet_mask = ~(mobile_mask | desktop_mask)

    if mobile_mask.any():
        mobile_probs = np.array(
            [0.015, 0.012, 0.010, 0.010, 0.010, 0.012, 0.020, 0.028, 0.040, 0.050, 0.055, 0.060,
             0.060, 0.055, 0.055, 0.060, 0.065, 0.075, 0.085, 0.085, 0.070, 0.055, 0.035, 0.021],
            dtype=float,
        )
        mobile_probs = mobile_probs / mobile_probs.sum()
        hours[mobile_mask] = rng.choice(np.arange(24), size=int(mobile_mask.sum()), p=mobile_probs)
    if desktop_mask.any():
        desktop_probs = np.array(
            [0.004, 0.003, 0.002, 0.002, 0.002, 0.003, 0.010, 0.025, 0.055, 0.085, 0.095, 0.095,
             0.090, 0.085, 0.080, 0.075, 0.070, 0.060, 0.050, 0.040, 0.030, 0.020, 0.012, 0.005],
            dtype=float,
        )
        desktop_probs = desktop_probs / desktop_probs.sum()
        hours[desktop_mask] = rng.choice(np.arange(24), size=int(desktop_mask.sum()), p=desktop_probs)
    if tablet_mask.any():
        tablet_probs = np.array(
            [0.006, 0.005, 0.004, 0.004, 0.004, 0.006, 0.015, 0.028, 0.045, 0.060, 0.070, 0.075,
             0.078, 0.075, 0.070, 0.068, 0.070, 0.075, 0.080, 0.076, 0.060, 0.040, 0.022, 0.009],
            dtype=float,
        )
        tablet_probs = tablet_probs / tablet_probs.sum()
        hours[tablet_mask] = rng.choice(np.arange(24), size=int(tablet_mask.sum()), p=tablet_probs)

    minutes = rng.integers(0, 60, size=n)
    return hours * 60 + minutes


def _build_event_frame(
    customer_ids: np.ndarray,
    timestamps: np.ndarray,
    event_type: str,
    session_ids: np.ndarray,
    rng: np.random.Generator,
    item_category: Optional[np.ndarray] = None,
    quantity: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    n = len(customer_ids)
    if n == 0:
        return _empty_event_frame()

    seeds = rng.integers(10_000_000, 99_999_999, size=n)
    return pd.DataFrame(
        {
            "event_id": [f"EVT-{event_type[:3].upper()}-{int(x)}" for x in seeds],
            "customer_id": customer_ids.astype(int),
            "timestamp": pd.to_datetime(timestamps),
            "event_type": event_type,
            "session_id": session_ids.astype(object),
            "item_category": item_category if item_category is not None else None,
            "quantity": quantity if quantity is not None else None,
        }
    )


def simulate_events(
    customers: pd.DataFrame,
    assignments: pd.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate the full customer journey and return raw tables:
    - events
    - orders
    - campaign_exposures
    - state_snapshots
    - final_state_metrics
    """
    rng = rng or np.random.default_rng(config.random_seed)

    sim = customers.merge(assignments, on="customer_id", how="left").sort_values("customer_id").reset_index(drop=True)
    tracker = StateTracker(n_customers=len(sim), coupon_fatigue_decay=config.coupon_fatigue_decay)
    coupon_cost_lookup = sim["coupon_cost"].to_numpy()

    event_frames: List[pd.DataFrame] = []
    order_frames: List[pd.DataFrame] = []
    exposure_frames: List[pd.DataFrame] = []
    snapshot_frames: List[pd.DataFrame] = []

    order_seq = 1
    exposure_seq = 1

    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    signup_dates = pd.to_datetime(sim["signup_date"]).to_numpy()
    assigned_at_days = (pd.to_datetime(sim["assigned_at"]) - pd.Timestamp(config.start_date)).dt.days.to_numpy()

    for day_idx, date in enumerate(dates):
        active_mask = signup_dates <= np.datetime64(date)
        tracker.start_day(active_mask)

        eligible_exposure = (
            active_mask
            & (sim["treatment_flag"].to_numpy() == 1)
            & (tracker.days_since_last_coupon >= config.coupon_cooldown_days)
            & (tracker.exposures_total < config.max_exposures_per_customer)
            & (
                (tracker.inactivity_days >= config.coupon_trigger_inactivity_days)
                | (assigned_at_days == day_idx)
            )
        )
        fatigue_penalty = np.clip(
            (tracker.coupon_fatigue_score / max(config.coupon_fatigue_guardrail, 1e-6))
            * sim.get("discount_fatigue_sensitivity", pd.Series(0.0, index=sim.index)).to_numpy(dtype=float),
            0.0,
            1.5,
        )
        brand_penalty = sim.get("brand_sensitivity", pd.Series(0.0, index=sim.index)).to_numpy(dtype=float)
        exposure_prob = np.clip(
            0.18 + 0.30 * sim["coupon_affinity"].to_numpy() + 0.10 * (tracker.inactivity_days >= config.coupon_trigger_inactivity_days)
            - 0.20 * fatigue_penalty
            - 0.08 * brand_penalty * fatigue_penalty,
            0.0,
            0.95,
        )
        exposure_mask = eligible_exposure & (rng.random(len(sim)) < exposure_prob)
        tracker.record_exposure(exposure_mask)

        if exposure_mask.any():
            exposure_ids = [f"EXP-{day_idx:03d}-{exposure_seq + i:07d}" for i in range(int(exposure_mask.sum()))]
            exposure_seq += int(exposure_mask.sum())
            exposure_times = pd.Timestamp(date.normalize()) + pd.to_timedelta(rng.integers(9 * 60, 21 * 60, size=int(exposure_mask.sum())), unit="m")
            exposure_frames.append(
                pd.DataFrame(
                    {
                        "exposure_id": exposure_ids,
                        "customer_id": sim.loc[exposure_mask, "customer_id"].to_numpy().astype(int),
                        "exposure_time": exposure_times.to_numpy(),
                        "campaign_type": sim.loc[exposure_mask, "campaign_type"].to_numpy(),
                        "coupon_cost": sim.loc[exposure_mask, "coupon_cost"].to_numpy().astype(int),
                    }
                )
            )

        coupon_open_prob = compute_coupon_open_probability(sim, exposure_mask, tracker)
        coupon_open_mask = exposure_mask & (rng.random(len(sim)) < coupon_open_prob)
        tracker.record_coupon_open(coupon_open_mask)

        visit_prob = compute_visit_probability(sim, tracker, active_mask, date)
        visit_mask = rng.random(len(sim)) < visit_prob
        tracker.record_visit(visit_mask, day_idx)

        browse_prob = compute_browse_probability(sim, visit_mask, tracker)
        browse_mask = visit_mask & (rng.random(len(sim)) < browse_prob)

        search_prob = compute_search_probability(sim, visit_mask, tracker)
        search_mask = visit_mask & (rng.random(len(sim)) < search_prob)

        add_cart_prob = compute_add_to_cart_probability(sim, browse_mask, search_mask, tracker)
        add_to_cart_mask = browse_mask & (rng.random(len(sim)) < add_cart_prob)
        tracker.record_cart_add(add_to_cart_mask)

        purchase_prob = compute_purchase_probability(sim, visit_mask, add_to_cart_mask, coupon_open_mask, tracker)
        purchase_mask = visit_mask & (rng.random(len(sim)) < purchase_prob)

        coupon_redeem_prob = compute_coupon_redeem_probability(sim, coupon_open_mask, purchase_mask, tracker)
        coupon_redeem_mask = coupon_open_mask & purchase_mask & (rng.random(len(sim)) < coupon_redeem_prob)
        tracker.record_coupon_redeem(coupon_redeem_mask)

        remove_cart_prob = compute_remove_cart_probability(sim, add_to_cart_mask, purchase_mask, tracker)
        remove_cart_mask = add_to_cart_mask & (rng.random(len(sim)) < remove_cart_prob)
        tracker.record_cart_remove(remove_cart_mask)

        support_prob = np.clip(
            sim["support_contact_propensity"].to_numpy() + 0.05 * remove_cart_mask.astype(float) + 0.03 * (tracker.inactivity_days > 20),
            0.0,
            0.55,
        )
        support_mask = visit_mask & (rng.random(len(sim)) < support_prob)

        n_customers = len(sim)
        session_ids = np.full(n_customers, None, dtype=object)
        session_start = np.full(n_customers, np.datetime64("NaT"), dtype="datetime64[ns]")
        session_category = np.full(n_customers, None, dtype=object)
        session_quantity = np.zeros(n_customers, dtype=int)
        pageview_counts = np.zeros(n_customers, dtype=int)

        visit_idx = np.flatnonzero(visit_mask)
        if len(visit_idx):
            visit_customers = sim.iloc[visit_idx]
            session_seed = rng.integers(10_000_000, 99_999_999, size=len(visit_idx))
            session_ids[visit_idx] = np.array([f"SES-{int(x)}" for x in session_seed], dtype=object)
            start_minutes = _sample_session_start_minutes(visit_customers["device_type"].to_numpy(), rng)
            session_start[visit_idx] = (
                pd.Timestamp(date.normalize()) + pd.to_timedelta(start_minutes, unit="m")
            ).to_numpy(dtype="datetime64[ns]")
            session_category[visit_idx] = rng.choice(_ITEM_CATEGORIES, size=len(visit_idx), p=_CATEGORY_PROBS)
            sampled_qty = np.clip(
                np.round(rng.normal(visit_customers["basket_size_preference"].to_numpy(), 0.60)).astype(int),
                1,
                6,
            )
            session_quantity[visit_idx] = sampled_qty
            pageview_counts[visit_idx] = np.where(
                browse_mask[visit_idx],
                rng.integers(1, 5, size=len(visit_idx)),
                0,
            )

        event_frames.append(
            _build_event_frame(
                sim.loc[visit_mask, "customer_id"].to_numpy(),
                session_start[visit_mask],
                "visit",
                session_ids[visit_mask],
                rng,
            )
        )

        browse_idx = np.flatnonzero(browse_mask)
        if len(browse_idx):
            counts = pageview_counts[browse_idx]
            repeated_idx = np.repeat(browse_idx, counts)
            offsets = np.repeat(np.cumsum(np.r_[0, counts[:-1]]), counts)
            rank_in_session = np.arange(int(counts.sum())) - offsets
            page_times = session_start[repeated_idx] + pd.to_timedelta(
                1 + rank_in_session * 2 + rng.integers(0, 2, size=len(repeated_idx)),
                unit="m",
            )
            event_frames.append(
                _build_event_frame(
                    sim.iloc[repeated_idx]["customer_id"].to_numpy(),
                    page_times,
                    "page_view",
                    session_ids[repeated_idx],
                    rng,
                    item_category=session_category[repeated_idx],
                )
            )
        else:
            event_frames.append(_empty_event_frame())

        def _offset_rows(mask: np.ndarray, event_type: str, min_minute: int, max_minute: int, include_category: bool = True, include_quantity: bool = False) -> None:
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                event_frames.append(_empty_event_frame())
                return
            offsets = rng.integers(min_minute, max_minute + 1, size=len(idx))
            timestamps = session_start[idx] + pd.to_timedelta(offsets, unit="m")
            item_category = session_category[idx] if include_category else None
            quantity = session_quantity[idx] if include_quantity else None
            event_frames.append(
                _build_event_frame(
                    sim.iloc[idx]["customer_id"].to_numpy(),
                    timestamps,
                    event_type,
                    session_ids[idx],
                    rng,
                    item_category=item_category,
                    quantity=quantity,
                )
            )

        _offset_rows(search_mask, "search", 2, 10, include_category=True, include_quantity=False)
        _offset_rows(add_to_cart_mask, "add_to_cart", 5, 16, include_category=True, include_quantity=True)
        _offset_rows(remove_cart_mask, "remove_from_cart", 8, 24, include_category=True, include_quantity=True)

        # Coupon open can happen without a site visit. For non-visit opens, create a lightweight standalone session.
        coupon_open_idx = np.flatnonzero(coupon_open_mask)
        if len(coupon_open_idx):
            standalone_mask = coupon_open_mask & ~visit_mask
            standalone_idx = np.flatnonzero(standalone_mask)
            if len(standalone_idx):
                session_seed = rng.integers(10_000_000, 99_999_999, size=len(standalone_idx))
                session_ids[standalone_idx] = np.array([f"SES-{int(x)}" for x in session_seed], dtype=object)
                session_start[standalone_idx] = (
                    pd.Timestamp(date.normalize()) + pd.to_timedelta(rng.integers(8 * 60, 23 * 60, size=len(standalone_idx)), unit="m")
                ).to_numpy(dtype="datetime64[ns]")
                session_category[standalone_idx] = rng.choice(_ITEM_CATEGORIES, size=len(standalone_idx), p=_CATEGORY_PROBS)
                session_quantity[standalone_idx] = np.clip(
                    np.round(rng.normal(sim.iloc[standalone_idx]["basket_size_preference"].to_numpy(), 0.60)).astype(int),
                    1,
                    6,
                )
            coupon_offsets = rng.integers(1, 12, size=len(coupon_open_idx))
            coupon_times = session_start[coupon_open_idx] + pd.to_timedelta(coupon_offsets, unit="m")
            event_frames.append(
                _build_event_frame(
                    sim.iloc[coupon_open_idx]["customer_id"].to_numpy(),
                    coupon_times,
                    "coupon_open",
                    session_ids[coupon_open_idx],
                    rng,
                    item_category=session_category[coupon_open_idx],
                )
            )
        else:
            event_frames.append(_empty_event_frame())

        _offset_rows(support_mask, "support_contact", 6, 25, include_category=False, include_quantity=False)

        purchase_idx = np.flatnonzero(purchase_mask)
        purchase_times = np.array([], dtype="datetime64[ns]")
        if len(purchase_idx):
            purchase_offsets = rng.integers(10, 28, size=len(purchase_idx))
            purchase_times = session_start[purchase_idx] + pd.to_timedelta(purchase_offsets, unit="m")
            event_frames.append(
                _build_event_frame(
                    sim.iloc[purchase_idx]["customer_id"].to_numpy(),
                    purchase_times,
                    "purchase",
                    session_ids[purchase_idx],
                    rng,
                    item_category=session_category[purchase_idx],
                    quantity=session_quantity[purchase_idx],
                )
            )
        else:
            event_frames.append(_empty_event_frame())

        if len(purchase_idx):
            redeem_offsets = rng.integers(11, 30, size=int(coupon_redeem_mask[purchase_idx].sum()))
            redeem_idx = purchase_idx[coupon_redeem_mask[purchase_idx]]
            redeem_times = session_start[redeem_idx] + pd.to_timedelta(redeem_offsets, unit="m")
            event_frames.append(
                _build_event_frame(
                    sim.iloc[redeem_idx]["customer_id"].to_numpy(),
                    redeem_times,
                    "coupon_redeem",
                    session_ids[redeem_idx],
                    rng,
                    item_category=session_category[redeem_idx],
                    quantity=session_quantity[redeem_idx],
                )
            )
        else:
            event_frames.append(_empty_event_frame())

        orders = build_orders(
            customers=sim,
            purchase_mask=purchase_mask,
            date=date,
            day_idx=day_idx,
            order_sequence_start=order_seq,
            coupon_open_mask=coupon_redeem_mask,
            coupon_cost_lookup=coupon_cost_lookup,
            rng=rng,
            item_categories=session_category[purchase_mask] if purchase_mask.any() else None,
            quantities=session_quantity[purchase_mask] if purchase_mask.any() else None,
            order_times=purchase_times if len(purchase_idx) else None,
        )
        if not orders.empty:
            order_seq += len(orders)
            order_frames.append(orders)
            amount_lookup = orders.set_index("customer_id")["net_amount"].to_dict()
            order_amounts = np.zeros(len(sim), dtype=float)
            purchase_idx = np.flatnonzero(purchase_mask)
            for idx in purchase_idx:
                cid = int(sim.iloc[idx]["customer_id"])
                order_amounts[idx] = amount_lookup.get(cid, 0.0)
            tracker.record_purchase(purchase_mask, order_amounts, day_idx, coupon_redeem_mask=coupon_redeem_mask)

        if (day_idx % config.snapshot_frequency_days == 0) or (day_idx == len(dates) - 1):
            snapshot_frames.append(
                tracker.to_snapshot(
                    customers=sim[["customer_id"]],
                    snapshot_date=date,
                    day_idx=day_idx,
                    dormant_threshold=config.dormant_inactivity_days,
                    churn_threshold=config.churn_inactivity_days,
                )
            )

    events = pd.concat([df for df in event_frames if not df.empty], ignore_index=True) if event_frames else pd.DataFrame()
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else pd.DataFrame()
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    state_snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    final_state = tracker.final_metrics(simulation_days=config.simulation_days)

    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"])
    if not orders.empty:
        orders["order_time"] = pd.to_datetime(orders["order_time"])
    if not exposures.empty:
        exposures["exposure_time"] = pd.to_datetime(exposures["exposure_time"])
    if not state_snapshots.empty:
        state_snapshots["snapshot_date"] = pd.to_datetime(state_snapshots["snapshot_date"])
        state_snapshots["last_visit_date"] = pd.to_datetime(state_snapshots["last_visit_date"])
        state_snapshots["last_purchase_date"] = pd.to_datetime(state_snapshots["last_purchase_date"])

    return events, orders, exposures, state_snapshots, final_state
