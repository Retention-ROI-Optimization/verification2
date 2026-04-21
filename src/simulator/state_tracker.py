from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .event_rules import classify_customer_status


@dataclass
class StateTracker:
    """
    Holds mutable simulation state in vectorized numpy arrays.
    """

    n_customers: int
    coupon_fatigue_decay: float = 0.94

    def __post_init__(self) -> None:
        n = self.n_customers
        self.last_visit_day = np.full(n, -1, dtype=int)
        self.last_purchase_day = np.full(n, -1, dtype=int)

        self.visits_total = np.zeros(n, dtype=int)
        self.purchases_total = np.zeros(n, dtype=int)
        self.monetary_total = np.zeros(n, dtype=float)

        self.cart_balance = np.zeros(n, dtype=int)

        self.exposures_total = np.zeros(n, dtype=int)
        self.coupon_open_total = np.zeros(n, dtype=int)
        self.coupon_redeemed_total = np.zeros(n, dtype=int)

        self.days_since_last_coupon = np.full(n, 999, dtype=int)
        self.inactivity_days = np.zeros(n, dtype=int)

        self.recent_visit_score = np.zeros(n, dtype=float)
        self.recent_purchase_score = np.zeros(n, dtype=float)
        self.recent_exposure_score = np.zeros(n, dtype=float)
        self.recent_cart_abandon_score = np.zeros(n, dtype=float)
        self.coupon_fatigue_score = np.zeros(n, dtype=float)
        self.discount_dependency_score = np.zeros(n, dtype=float)

    def start_day(self, active_mask: np.ndarray | None = None) -> None:
        if active_mask is None:
            active_mask = np.ones(self.n_customers, dtype=bool)

        if np.any(active_mask):
            self.inactivity_days[active_mask] += 1
            self.days_since_last_coupon[active_mask] += 1

            self.recent_visit_score[active_mask] *= 0.86
            self.recent_purchase_score[active_mask] *= 0.91
            self.recent_exposure_score[active_mask] *= 0.68
            self.recent_cart_abandon_score[active_mask] *= 0.88
            self.coupon_fatigue_score[active_mask] *= self.coupon_fatigue_decay
            self.discount_dependency_score[active_mask] *= 0.96

    def record_exposure(self, mask: np.ndarray) -> None:
        if mask.any():
            self.exposures_total[mask] += 1
            self.days_since_last_coupon[mask] = 0
            self.recent_exposure_score[mask] += 1.0
            self.coupon_fatigue_score[mask] += 1.0

    def record_coupon_open(self, mask: np.ndarray) -> None:
        if mask.any():
            self.coupon_open_total[mask] += 1

    def record_coupon_redeem(self, mask: np.ndarray) -> None:
        if mask.any():
            self.coupon_redeemed_total[mask] += 1
            self.coupon_fatigue_score[mask] += 0.35
            self.discount_dependency_score[mask] += 0.55

    def record_visit(self, mask: np.ndarray, day_idx: int) -> None:
        if mask.any():
            self.last_visit_day[mask] = day_idx
            self.visits_total[mask] += 1
            self.inactivity_days[mask] = 0
            self.recent_visit_score[mask] += 1.0

    def record_cart_add(self, mask: np.ndarray) -> None:
        if mask.any():
            self.cart_balance[mask] += 1

    def record_cart_remove(self, mask: np.ndarray) -> None:
        if mask.any():
            self.cart_balance[mask] = np.maximum(self.cart_balance[mask] - 1, 0)
            self.recent_cart_abandon_score[mask] += 1.0

    def record_purchase(
        self,
        mask: np.ndarray,
        order_amounts: np.ndarray,
        day_idx: int,
        coupon_redeem_mask: np.ndarray | None = None,
    ) -> None:
        if not mask.any():
            return
        self.last_purchase_day[mask] = day_idx
        self.purchases_total[mask] += 1
        self.monetary_total[mask] += order_amounts[mask]
        self.inactivity_days[mask] = 0
        self.recent_purchase_score[mask] += 1.5
        self.cart_balance[mask] = np.maximum(self.cart_balance[mask] - 1, 0)

        if coupon_redeem_mask is None:
            coupon_redeem_mask = np.zeros(self.n_customers, dtype=bool)
        coupon_redeem_mask = coupon_redeem_mask & mask
        organic_purchase_mask = mask & ~coupon_redeem_mask
        if organic_purchase_mask.any():
            self.coupon_fatigue_score[organic_purchase_mask] = np.maximum(
                self.coupon_fatigue_score[organic_purchase_mask] - 0.35,
                0.0,
            )
            self.discount_dependency_score[organic_purchase_mask] = np.maximum(
                self.discount_dependency_score[organic_purchase_mask] - 0.25,
                0.0,
            )
        if coupon_redeem_mask.any():
            self.discount_dependency_score[coupon_redeem_mask] += 0.20

    def to_snapshot(
        self,
        customers: pd.DataFrame,
        snapshot_date: pd.Timestamp,
        day_idx: int,
        dormant_threshold: int,
        churn_threshold: int,
    ) -> pd.DataFrame:
        last_visit_date = np.where(
            self.last_visit_day >= 0,
            (pd.Timestamp(snapshot_date.normalize()) - pd.to_timedelta(day_idx - self.last_visit_day, unit="D")).astype("datetime64[ns]"),
            np.datetime64("NaT"),
        )
        last_purchase_date = np.where(
            self.last_purchase_day >= 0,
            (pd.Timestamp(snapshot_date.normalize()) - pd.to_timedelta(day_idx - self.last_purchase_day, unit="D")).astype("datetime64[ns]"),
            np.datetime64("NaT"),
        )

        status = classify_customer_status(
            inactivity_days=self.inactivity_days,
            dormant_threshold=dormant_threshold,
            churn_threshold=churn_threshold,
        )

        return pd.DataFrame(
            {
                "customer_id": customers["customer_id"].to_numpy(),
                "snapshot_date": pd.Timestamp(snapshot_date.normalize()),
                "last_visit_date": pd.to_datetime(last_visit_date),
                "last_purchase_date": pd.to_datetime(last_purchase_date),
                "visits_total": self.visits_total.astype(int),
                "purchases_total": self.purchases_total.astype(int),
                "monetary_total": self.monetary_total.astype(float),
                "inactivity_days": self.inactivity_days.astype(int),
                "current_status": status,
                "recent_visit_score": self.recent_visit_score.astype(float),
                "recent_purchase_score": self.recent_purchase_score.astype(float),
                "recent_exposure_score": self.recent_exposure_score.astype(float),
                "coupon_fatigue_score": self.coupon_fatigue_score.astype(float),
                "discount_dependency_score": self.discount_dependency_score.astype(float),
            }
        )

    def final_metrics(self, simulation_days: int) -> pd.DataFrame:
        recency_days = np.where(
            self.last_purchase_day >= 0,
            simulation_days - 1 - self.last_purchase_day,
            np.where(self.last_visit_day >= 0, simulation_days - 1 - self.last_visit_day, simulation_days),
        )

        return pd.DataFrame(
            {
                "customer_id": np.arange(1, self.n_customers + 1, dtype=int),
                "last_visit_day": self.last_visit_day,
                "last_purchase_day": self.last_purchase_day,
                "recency_days": recency_days.astype(int),
                "frequency": self.purchases_total.astype(int),
                "monetary": self.monetary_total.astype(float),
                "coupon_exposures": self.exposures_total.astype(int),
                "coupon_opens": self.coupon_open_total.astype(int),
                "coupon_redeemed": self.coupon_redeemed_total.astype(int),
                "inactivity_days": self.inactivity_days.astype(int),
                "coupon_fatigue_score": self.coupon_fatigue_score.astype(float),
                "discount_dependency_score": self.discount_dependency_score.astype(float),
            }
        )
