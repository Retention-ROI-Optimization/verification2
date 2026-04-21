from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import SimulationConfig


def assign_treatment(
    customers: pd.DataFrame,
    config: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Balanced treatment/control assignment.

    Assignment is stratified by persona and latent uplift segment so that
    later uplift comparisons remain balanced without over-fragmenting the data
    into tiny month-level strata.
    """
    rng = rng or np.random.default_rng(config.random_seed)

    df = customers[
        [
            "customer_id",
            "persona",
            "uplift_segment_true",
            "acquisition_month",
            "signup_date",
            "price_sensitivity",
            "coupon_affinity",
        ]
    ].copy()
    df["treatment_flag"] = 0

    if config.stratify_treatment:
        for _, idx in df.groupby(["persona", "uplift_segment_true"]).groups.items():
            idx = np.array(list(idx), dtype=int)
            rng.shuffle(idx)
            treated_n = int(round(len(idx) * config.treatment_share))
            df.loc[idx[:treated_n], "treatment_flag"] = 1
    else:
        treated_n = int(round(len(df) * config.treatment_share))
        shuffled = rng.permutation(df.index.to_numpy())
        df.loc[shuffled[:treated_n], "treatment_flag"] = 1

    df["treatment_group"] = np.where(df["treatment_flag"] == 1, "treatment", "control")

    base_coupon = (
        config.coupon_min_cost
        + (
            0.55 * df["price_sensitivity"]
            + 0.45 * df["coupon_affinity"]
        ) * (config.coupon_max_cost - config.coupon_min_cost)
    )
    noise = rng.normal(0, 600, size=len(df))
    coupon_cost = np.clip(base_coupon + noise, config.coupon_min_cost, config.coupon_max_cost).round().astype(int)

    assigned_offset_days = rng.integers(0, 7, size=len(df))
    assigned_at = pd.to_datetime(df["signup_date"]) + pd.to_timedelta(assigned_offset_days, unit="D")

    assignments = pd.DataFrame(
        {
            "customer_id": df["customer_id"].astype(int),
            "treatment_group": df["treatment_group"],
            "treatment_flag": df["treatment_flag"].astype(int),
            "campaign_type": config.campaign_type,
            "coupon_cost": coupon_cost,
            "assigned_at": assigned_at,
        }
    )

    return assignments.sort_values("customer_id").reset_index(drop=True)
