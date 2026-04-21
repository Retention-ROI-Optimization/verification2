from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CustomerRecord:
    customer_id: int
    persona: str
    uplift_segment_true: str
    signup_date: str
    acquisition_month: str
    region: str
    device_type: str
    acquisition_channel: str


@dataclass(frozen=True)
class TreatmentAssignmentRecord:
    customer_id: int
    treatment_group: str
    treatment_flag: int
    campaign_type: str
    coupon_cost: int
    assigned_at: str


@dataclass(frozen=True)
class CampaignExposureRecord:
    exposure_id: str
    customer_id: int
    exposure_time: str
    campaign_type: str
    coupon_cost: int


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    customer_id: int
    timestamp: str
    event_type: str
    session_id: str
    item_category: Optional[str] = None
    quantity: Optional[int] = None


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    customer_id: int
    order_time: str
    item_category: str
    quantity: int
    gross_amount: float
    discount_amount: float
    net_amount: float
    coupon_used: int


@dataclass(frozen=True)
class StateSnapshotRecord:
    customer_id: int
    snapshot_date: str
    last_visit_date: Optional[str]
    last_purchase_date: Optional[str]
    visits_total: int
    purchases_total: int
    monetary_total: float
    inactivity_days: int
    current_status: str
