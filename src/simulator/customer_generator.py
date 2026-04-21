from __future__ import annotations

import calendar
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .personas import (
    DEFAULT_PERSONAS,
    DEFAULT_UPLIFT_SEGMENTS,
    PERSONA_TO_UPLIFT_WEIGHTS,
    PersonaProfile,
)


_PERSONA_SIGNUP_MONTH_WEIGHTS = {
    "new_signup": np.array([1, 1, 1, 1, 2, 2, 3, 5, 8, 10, 10, 8], dtype=float),
    "churn_progressing": np.array([8, 8, 8, 7, 7, 6, 5, 4, 3, 2, 2, 2], dtype=float),
}


def _month_start(month_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month_str}-01")


def _random_signup_dates(
    persona: np.ndarray,
    rng: np.random.Generator,
    signup_months,
) -> tuple[pd.Series, pd.Series]:
    month_choices = np.empty(len(persona), dtype=object)

    for persona_name in np.unique(persona):
        mask = persona == persona_name
        count = int(mask.sum())
        if count == 0:
            continue

        weights = _PERSONA_SIGNUP_MONTH_WEIGHTS.get(persona_name)
        if weights is None:
            chosen = rng.choice(signup_months, size=count, replace=True)
        else:
            normalized = weights / weights.sum()
            chosen = rng.choice(signup_months, size=count, replace=True, p=normalized)
        month_choices[mask] = chosen

    offsets = []
    for month_str in month_choices:
        year, month = map(int, month_str.split("-"))
        days_in_month = calendar.monthrange(year, month)[1]
        offsets.append(rng.integers(0, days_in_month))

    signup_dates = [
        _month_start(m) + pd.Timedelta(days=int(off))
        for m, off in zip(month_choices, offsets)
    ]
    return pd.Series(signup_dates), pd.Series(month_choices)


def generate_customers(
    config: SimulationConfig,
    personas: Optional[Dict[str, PersonaProfile]] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate customer master data.

    Output is intentionally rich enough for both:
    - raw simulator use
    - later customer-level feature aggregation
    """
    personas = personas or DEFAULT_PERSONAS
    rng = rng or np.random.default_rng(config.random_seed)

    persona_names = list(personas.keys())
    persona_weights = np.array([personas[name].acquisition_weight for name in persona_names], dtype=float)
    persona_weights = persona_weights / persona_weights.sum()

    n = config.n_customers
    customer_ids = np.arange(1, n + 1, dtype=int)
    persona = rng.choice(persona_names, size=n, p=persona_weights)

    uplift_segment = np.empty(n, dtype=object)
    uplift_names = list(DEFAULT_UPLIFT_SEGMENTS.keys())
    for persona_name in persona_names:
        mask = persona == persona_name
        count = int(mask.sum())
        if count == 0:
            continue

        uplift_weights = np.array(
            [PERSONA_TO_UPLIFT_WEIGHTS[persona_name][name] for name in uplift_names],
            dtype=float,
        )
        uplift_weights = uplift_weights / uplift_weights.sum()
        uplift_segment[mask] = rng.choice(uplift_names, size=count, p=uplift_weights)

    signup_date, acquisition_month = _random_signup_dates(
        persona=persona,
        rng=rng,
        signup_months=config.signup_months,
    )

    region = rng.choice(
        ["Seoul", "Busan", "Incheon", "Daejeon", "Daegu", "Gwangju"],
        size=n,
        p=[0.34, 0.18, 0.13, 0.10, 0.15, 0.10],
    )
    device_type = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n,
        p=[0.62, 0.30, 0.08],
    )
    acquisition_channel = rng.choice(
        ["organic", "paid_ads", "referral", "email", "social"],
        size=n,
        p=[0.28, 0.24, 0.14, 0.17, 0.17],
    )

    base_visit_prob = np.zeros(n, dtype=float)
    browse_prob_base = np.zeros(n, dtype=float)
    search_prob_base = np.zeros(n, dtype=float)
    add_to_cart_prob_base = np.zeros(n, dtype=float)
    remove_from_cart_prob_base = np.zeros(n, dtype=float)
    purchase_given_cart_base = np.zeros(n, dtype=float)
    purchase_given_visit_base = np.zeros(n, dtype=float)
    coupon_open_prob_base = np.zeros(n, dtype=float)
    coupon_redeem_prob_base = np.zeros(n, dtype=float)
    avg_order_value_mean = np.zeros(n, dtype=float)
    avg_order_value_std = np.zeros(n, dtype=float)
    churn_sensitivity_base = np.zeros(n, dtype=float)
    price_sensitivity = np.zeros(n, dtype=float)
    recovery_prob_base = np.zeros(n, dtype=float)
    treatment_lift_base = np.zeros(n, dtype=float)
    discount_fatigue_sensitivity = np.zeros(n, dtype=float)
    offer_dependency_risk = np.zeros(n, dtype=float)
    brand_sensitivity = np.zeros(n, dtype=float)

    for persona_name, profile in personas.items():
        mask = persona == persona_name
        count = int(mask.sum())
        if count == 0:
            continue

        base_visit_prob[mask] = np.clip(rng.normal(profile.visit_prob, 0.03, size=count), 0.05, 0.78)
        browse_prob_base[mask] = np.clip(rng.normal(profile.browse_prob, 0.05, size=count), 0.25, 0.96)
        search_prob_base[mask] = np.clip(rng.normal(profile.search_prob, 0.05, size=count), 0.05, 0.92)
        add_to_cart_prob_base[mask] = np.clip(rng.normal(profile.add_to_cart_prob, 0.04, size=count), 0.05, 0.82)
        remove_from_cart_prob_base[mask] = np.clip(rng.normal(profile.remove_from_cart_prob, 0.03, size=count), 0.01, 0.72)
        purchase_given_cart_base[mask] = np.clip(rng.normal(profile.purchase_given_cart_prob, 0.05, size=count), 0.05, 0.95)
        purchase_given_visit_base[mask] = np.clip(rng.normal(profile.purchase_given_visit_prob, 0.02, size=count), 0.01, 0.32)
        coupon_open_prob_base[mask] = np.clip(rng.normal(profile.coupon_open_prob, 0.05, size=count), 0.01, 0.95)
        coupon_redeem_prob_base[mask] = np.clip(rng.normal(profile.coupon_redeem_prob, 0.05, size=count), 0.01, 0.90)
        avg_order_value_mean[mask] = np.clip(rng.normal(profile.avg_order_mean, profile.avg_order_std * 0.25, size=count), 25000, None)
        avg_order_value_std[mask] = np.clip(rng.normal(profile.avg_order_std, profile.avg_order_std * 0.15, size=count), 6000, None)
        churn_sensitivity_base[mask] = np.clip(rng.normal(profile.churn_sensitivity, 0.10, size=count), 0.40, 1.90)
        price_sensitivity[mask] = np.clip(rng.normal(profile.price_sensitivity, 0.08, size=count), 0.05, 0.98)
        recovery_prob_base[mask] = np.clip(rng.normal(profile.recovery_prob, 0.05, size=count), 0.01, 0.82)

        fatigue_base = 0.35 + 0.25 * price_sensitivity[mask] + 0.12 * churn_sensitivity_base[mask]
        dependency_base = 0.22 + 0.38 * coupon_redeem_prob_base[mask] + 0.14 * price_sensitivity[mask]
        brand_base = 0.20 + 0.28 * price_sensitivity[mask] + 0.10 * (persona_name in {"vip_loyal", "regular_loyal"})
        if persona_name in {"price_sensitive", "explorer"}:
            fatigue_base += 0.10
            dependency_base += 0.08
        if persona_name == "new_signup":
            brand_base += 0.06
        discount_fatigue_sensitivity[mask] = np.clip(rng.normal(fatigue_base, 0.06, size=count), 0.08, 0.98)
        offer_dependency_risk[mask] = np.clip(rng.normal(dependency_base, 0.07, size=count), 0.05, 0.98)
        brand_sensitivity[mask] = np.clip(rng.normal(brand_base, 0.05, size=count), 0.05, 0.95)

    for segment_name, profile in DEFAULT_UPLIFT_SEGMENTS.items():
        mask = uplift_segment == segment_name
        count = int(mask.sum())
        if count == 0:
            continue

        treatment_lift_base[mask] = np.clip(
            rng.normal(profile.treatment_lift, 0.025, size=count),
            -0.18,
            0.42,
        )
        coupon_open_prob_base[mask] = np.clip(
            coupon_open_prob_base[mask] + rng.normal(profile.coupon_open_delta, 0.015, size=count),
            0.01,
            0.95,
        )
        coupon_redeem_prob_base[mask] = np.clip(
            coupon_redeem_prob_base[mask] + rng.normal(profile.coupon_redeem_delta, 0.015, size=count),
            0.01,
            0.92,
        )

    coupon_affinity = np.clip(
        0.55 * coupon_open_prob_base + 0.45 * coupon_redeem_prob_base + rng.normal(0, 0.04, size=n),
        0.02,
        0.98,
    )
    basket_size_preference = np.clip(
        rng.normal(1.4 + 2.0 * (avg_order_value_mean / max(avg_order_value_mean.max(), 1)), 0.35, size=n),
        1.0,
        5.0,
    )
    support_contact_propensity = np.clip(
        rng.normal(0.05 + 0.10 * price_sensitivity + 0.07 * churn_sensitivity_base, 0.03, size=n),
        0.01,
        0.45,
    )

    segment_fatigue_adjust = np.select(
        [
            uplift_segment == "persuadable",
            uplift_segment == "sure_thing",
            uplift_segment == "lost_cause",
            uplift_segment == "sleeping_dog",
        ],
        [0.04, 0.02, 0.08, 0.10],
        default=0.0,
    )
    discount_fatigue_sensitivity = np.clip(discount_fatigue_sensitivity + segment_fatigue_adjust, 0.08, 0.98)
    offer_dependency_risk = np.clip(offer_dependency_risk + 0.10 * coupon_affinity, 0.05, 0.98)
    brand_sensitivity = np.clip(brand_sensitivity + 0.08 * (coupon_affinity < 0.20), 0.05, 0.95)

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "persona": persona,
            "uplift_segment_true": uplift_segment,
            "signup_date": pd.to_datetime(signup_date),
            "acquisition_month": acquisition_month.astype(str),
            "region": region,
            "device_type": device_type,
            "acquisition_channel": acquisition_channel,
            "base_visit_prob": base_visit_prob,
            "browse_prob_base": browse_prob_base,
            "search_prob_base": search_prob_base,
            "add_to_cart_prob_base": add_to_cart_prob_base,
            "remove_from_cart_prob_base": remove_from_cart_prob_base,
            "purchase_given_cart_base": purchase_given_cart_base,
            "purchase_given_visit_base": purchase_given_visit_base,
            "coupon_open_prob_base": coupon_open_prob_base,
            "coupon_redeem_prob_base": coupon_redeem_prob_base,
            "avg_order_value_mean": avg_order_value_mean,
            "avg_order_value_std": avg_order_value_std,
            "churn_sensitivity_base": churn_sensitivity_base,
            "price_sensitivity": price_sensitivity,
            "coupon_affinity": coupon_affinity,
            "recovery_prob_base": recovery_prob_base,
            "treatment_lift_base": treatment_lift_base,
            "discount_fatigue_sensitivity": discount_fatigue_sensitivity,
            "offer_dependency_risk": offer_dependency_risk,
            "brand_sensitivity": brand_sensitivity,
            "basket_size_preference": basket_size_preference,
            "support_contact_propensity": support_contact_propensity,
        }
    )

    customers["days_from_simulation_start"] = (
        customers["signup_date"] - pd.Timestamp(config.start_date)
    ).dt.days.astype(int)

    return customers.sort_values("customer_id").reset_index(drop=True)
