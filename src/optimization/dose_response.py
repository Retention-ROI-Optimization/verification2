from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DOSE_RESPONSE_MODEL_VERSION = "dr_learner_v1"
CONTROL_LABEL = "control"
ACTION_INTENSITIES = ("low", "mid", "high")
DOSE_LEVELS = (CONTROL_LABEL,) + ACTION_INTENSITIES
DEFAULT_HORIZON_DAYS = 60
MIN_ARM_SAMPLES = 80


def _safe_series(values: Any, index: Optional[pd.Index] = None, default: float = 0.0) -> pd.Series:
    if values is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series([float(default)] * len(index), index=index, dtype=float)
    series = pd.to_numeric(pd.Series(values, index=index), errors="coerce").fillna(float(default))
    return series.astype(float)


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    numerator = np.asarray(a, dtype=float)
    denominator = np.asarray(b, dtype=float)
    return numerator / np.maximum(denominator, 1e-9)


def _normalize(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    low = float(np.nanmin(arr))
    high = float(np.nanmax(arr))
    if not np.isfinite(low) or not np.isfinite(high) or abs(high - low) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - low) / (high - low)


def _prepare_dates(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[column], errors="coerce").dt.normalize()


def _candidate_dirs(anchor: Optional[Path] = None) -> list[Path]:
    anchor = anchor or Path(__file__).resolve()
    project_root = anchor.resolve().parents[2]
    candidates = [Path.cwd(), project_root]
    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            out.append(path)
            seen.add(resolved)
    return out


def _resolve_default_paths(
    data_dir: Optional[str | Path] = None,
    model_dir: Optional[str | Path] = None,
    result_dir: Optional[str | Path] = None,
) -> tuple[Path, Path, Path]:
    candidates = _candidate_dirs()
    if data_dir is None:
        for root in candidates:
            path = root / "data" / "raw"
            if path.exists():
                data_dir = path
                break
        else:
            data_dir = candidates[0] / "data" / "raw"
    if model_dir is None:
        for root in candidates:
            path = root / "models"
            if path.exists():
                model_dir = path
                break
        else:
            model_dir = candidates[0] / "models"
    if result_dir is None:
        for root in candidates:
            path = root / "results"
            if path.exists():
                result_dir = path
                break
        else:
            result_dir = candidates[0] / "results"
    return Path(data_dir), Path(model_dir), Path(result_dir)


class ConstantProbabilityModel:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = float(np.clip(probability, 0.001, 0.999))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ConstantProbabilityModel":
        if y is not None and len(y):
            self.probability = float(np.clip(np.mean(pd.Series(y).astype(float)), 0.001, 0.999))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        p = np.clip(self.probability, 0.001, 0.999)
        return np.column_stack([np.full(n, 1.0 - p), np.full(n, p)])


class ConstantRegressor:
    def __init__(self, constant: float = 0.0) -> None:
        self.constant = float(constant)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ConstantRegressor":
        if y is not None and len(y):
            self.constant = float(np.mean(pd.Series(y).astype(float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.constant, dtype=float)


@dataclass
class DoseResponsePolicyModel:
    version: str
    numeric_features: list[str]
    categorical_features: list[str]
    propensity_model: Any
    outcome_models: Dict[str, Any]
    effect_models: Dict[str, Any]
    effect_priors: Dict[str, float]
    intensity_cost_multipliers: Dict[str, float]
    arm_summary: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.copy()
        if "tenure_days_at_assignment" not in X.columns:
            signup = pd.to_datetime(X.get("signup_date"), errors="coerce")
            assigned = pd.to_datetime(X.get("assigned_at"), errors="coerce")
            if signup.notna().any() and assigned.notna().any():
                X["tenure_days_at_assignment"] = (assigned.dt.normalize() - signup.dt.normalize()).dt.days.clip(lower=0)
            else:
                X["tenure_days_at_assignment"] = 0.0
        if "avg_order_value_hist" not in X.columns:
            monetary = _safe_series(X.get("monetary"), index=X.index, default=0.0)
            frequency = _safe_series(X.get("frequency"), index=X.index, default=0.0)
            X["avg_order_value_hist"] = _safe_div(monetary, np.maximum(frequency, 0.0))
        if "uplift_segment" not in X.columns:
            X["uplift_segment"] = "Unknown"
        for column in self.numeric_features:
            X[column] = _safe_series(X.get(column), index=X.index, default=0.0)
        for column in self.categorical_features:
            if column not in X.columns:
                X[column] = "Unknown"
            X[column] = X[column].fillna("Unknown").astype(str)
        return X[self.feature_columns].copy()

    def predict_retention_probability(self, frame: pd.DataFrame, dose_label: str) -> np.ndarray:
        X = self._prepare_features(frame)
        model = self.outcome_models[dose_label]
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return np.clip(proba.astype(float), 0.001, 0.999)
        return np.clip(proba[:, -1].astype(float), 0.001, 0.999)

    def predict_incremental_effect(self, frame: pd.DataFrame, intensity_key: str) -> np.ndarray:
        if intensity_key not in ACTION_INTENSITIES:
            return np.zeros(len(frame), dtype=float)
        X = self._prepare_features(frame)
        tau_hat = np.asarray(self.effect_models[intensity_key].predict(X), dtype=float)
        return np.clip(tau_hat, -0.35, 0.75)

    def predict_effect_frame(self, frame: pd.DataFrame, intensity_key: str) -> pd.DataFrame:
        uplift = self.predict_incremental_effect(frame, intensity_key)
        treated_prob = self.predict_retention_probability(frame, intensity_key)
        control_prob = self.predict_retention_probability(frame, CONTROL_LABEL)
        return pd.DataFrame(
            {
                "dose_response_incremental_effect": uplift,
                "dose_response_retention_prob_treated": treated_prob,
                "dose_response_retention_prob_control": control_prob,
            },
            index=frame.index,
        )


def _build_training_dataset(data_dir: Path, horizon_days: int = DEFAULT_HORIZON_DAYS) -> pd.DataFrame:
    customer_summary = pd.read_csv(data_dir / "customer_summary.csv")
    assignments = pd.read_csv(data_dir / "treatment_assignments.csv")
    orders = pd.read_csv(data_dir / "orders.csv")
    exposures_path = data_dir / "campaign_exposures.csv"
    exposures = pd.read_csv(exposures_path) if exposures_path.exists() else pd.DataFrame()

    df = customer_summary.copy()
    assignments = assignments.copy()
    assignments["assigned_at"] = _prepare_dates(assignments, "assigned_at")
    assignment_view = assignments[
        ["customer_id", "assigned_at", "treatment_flag", "treatment_group", "coupon_cost", "campaign_type"]
    ].copy()

    if "assigned_at" in df.columns:
        df["assigned_at"] = _prepare_dates(df, "assigned_at")
    df = df.drop(columns=[c for c in ["assigned_at", "treatment_flag", "treatment_group", "coupon_cost", "campaign_type"] if c in df.columns], errors="ignore")
    df = df.merge(assignment_view, on="customer_id", how="left")

    df["assigned_at"] = _prepare_dates(df, "assigned_at")
    df["signup_date"] = pd.to_datetime(df.get("signup_date"), errors="coerce").dt.normalize()
    df["tenure_days_at_assignment"] = (df["assigned_at"] - df["signup_date"]).dt.days.clip(lower=0)
    df["avg_order_value_hist"] = _safe_div(
        _safe_series(df.get("monetary"), index=df.index, default=0.0),
        np.maximum(_safe_series(df.get("frequency"), index=df.index, default=0.0), 0.0),
    )

    orders = orders.copy()
    orders["order_date"] = _prepare_dates(orders, "order_time")
    order_view = orders.merge(df[["customer_id", "assigned_at"]], on="customer_id", how="inner")
    order_diff = (order_view["order_date"] - order_view["assigned_at"]).dt.days
    post_orders = order_view[(order_diff >= 0) & (order_diff <= int(horizon_days))].copy()
    post_orders_count = post_orders.groupby("customer_id").size().rename("orders_post_horizon")
    post_revenue = post_orders.groupby("customer_id")["net_amount"].sum().rename("revenue_post_horizon")
    post_discount = post_orders.groupby("customer_id")["discount_amount"].sum().rename("discount_post_horizon")

    df = df.merge(post_orders_count, on="customer_id", how="left")
    df = df.merge(post_revenue, on="customer_id", how="left")
    df = df.merge(post_discount, on="customer_id", how="left")
    df["orders_post_horizon"] = _safe_series(df.get("orders_post_horizon"), index=df.index, default=0.0)
    df["revenue_post_horizon"] = _safe_series(df.get("revenue_post_horizon"), index=df.index, default=0.0)
    df["discount_post_horizon"] = _safe_series(df.get("discount_post_horizon"), index=df.index, default=0.0)
    df["retained_horizon"] = ((df["orders_post_horizon"] > 0) | (_safe_series(df.get("purchase_last_30"), index=df.index, default=0.0) > 0)).astype(int)

    if not exposures.empty:
        exposures = exposures.copy()
        exposures["exposure_date"] = _prepare_dates(exposures, "exposure_time")
        exposure_view = exposures.merge(df[["customer_id", "assigned_at"]], on="customer_id", how="inner")
        exposure_diff = (exposure_view["exposure_date"] - exposure_view["assigned_at"]).dt.days
        window_exposure = exposure_view[(exposure_diff >= 0) & (exposure_diff <= min(int(horizon_days), 30))].copy()
        exposure_count = window_exposure.groupby("customer_id").size().rename("dose_exposure_count")
    else:
        exposure_count = pd.Series(dtype=float, name="dose_exposure_count")
    df = df.merge(exposure_count, on="customer_id", how="left")
    df["dose_exposure_count"] = _safe_series(df.get("dose_exposure_count"), index=df.index, default=0.0)

    df["treatment_flag"] = _safe_series(df.get("treatment_flag"), index=df.index, default=0.0).astype(int)
    df["coupon_cost"] = _safe_series(df.get("coupon_cost"), index=df.index, default=0.0)

    treated_mask = df["treatment_flag"] == 1
    dose_labels = pd.Series(CONTROL_LABEL, index=df.index, dtype=object)
    if treated_mask.any():
        treated = df.loc[treated_mask, ["coupon_cost", "dose_exposure_count"]].copy()
        cost_rank = pd.Series(_normalize(treated["coupon_cost"].to_numpy()), index=treated.index)
        exposure_rank = pd.Series(_normalize(treated["dose_exposure_count"].to_numpy()), index=treated.index)
        dose_score = 0.75 * cost_rank + 0.25 * exposure_rank
        q1, q2 = np.quantile(dose_score.to_numpy(), [1 / 3, 2 / 3])
        treated_label = np.where(
            dose_score <= q1,
            "low",
            np.where(dose_score <= q2, "mid", "high"),
        )
        dose_labels.loc[treated.index] = treated_label
    df["dose_intensity"] = dose_labels.astype(str)

    feature_columns = [
        "recency_days",
        "frequency",
        "monetary",
        "visits_last_7",
        "visits_prev_7",
        "visit_change_rate",
        "purchase_last_30",
        "purchase_prev_30",
        "purchase_change_rate",
        "churn_probability",
        "uplift_score",
        "clv",
        "inactivity_days",
        "price_sensitivity",
        "coupon_affinity",
        "treatment_lift_base",
        "basket_size_preference",
        "support_contact_propensity",
        "tenure_days_at_assignment",
        "avg_order_value_hist",
        "persona",
        "region",
        "device_type",
        "acquisition_channel",
        "uplift_segment",
    ]
    for column in feature_columns:
        if column not in df.columns:
            if column in {"persona", "region", "device_type", "acquisition_channel", "uplift_segment"}:
                df[column] = "Unknown"
            else:
                df[column] = 0.0
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int)
    return df


def _feature_lists(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["persona", "region", "device_type", "acquisition_channel", "uplift_segment"]
    numeric = [
        "recency_days",
        "frequency",
        "monetary",
        "visits_last_7",
        "visits_prev_7",
        "visit_change_rate",
        "purchase_last_30",
        "purchase_prev_30",
        "purchase_change_rate",
        "churn_probability",
        "uplift_score",
        "clv",
        "inactivity_days",
        "price_sensitivity",
        "coupon_affinity",
        "treatment_lift_base",
        "basket_size_preference",
        "support_contact_propensity",
        "tenure_days_at_assignment",
        "avg_order_value_hist",
    ]
    numeric = [col for col in numeric if col in dataset.columns]
    categorical = [col for col in categorical if col in dataset.columns]
    return numeric, categorical


def _build_preprocessor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> ColumnTransformer:
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def _build_classifier(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> Pipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    model = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _build_regressor(numeric_features: Iterable[str], categorical_features: Iterable[str]) -> Pipeline:
    preprocessor = _build_preprocessor(numeric_features, categorical_features)
    model = Ridge(alpha=2.0, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _fit_binary_outcome_model(X: pd.DataFrame, y: pd.Series, numeric_features: list[str], categorical_features: list[str]) -> Any:
    y = pd.Series(y).astype(int)
    if len(X) < MIN_ARM_SAMPLES or y.nunique() < 2:
        return ConstantProbabilityModel(float(y.mean()) if len(y) else 0.5)
    model = _build_classifier(numeric_features, categorical_features)
    model.fit(X, y)
    return model


def _fit_effect_model(X: pd.DataFrame, tau: pd.Series, numeric_features: list[str], categorical_features: list[str]) -> Any:
    tau = pd.Series(tau).astype(float)
    if len(X) < MIN_ARM_SAMPLES:
        return ConstantRegressor(float(tau.mean()) if len(tau) else 0.0)
    model = _build_regressor(numeric_features, categorical_features)
    model.fit(X, tau)
    return model


def _predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return np.clip(np.asarray(proba, dtype=float), 0.001, 0.999)
        return np.clip(np.asarray(proba[:, -1], dtype=float), 0.001, 0.999)
    return np.clip(np.asarray(model.predict(X), dtype=float), 0.001, 0.999)


def _sample_training_dataset(dataset: pd.DataFrame, max_rows: int = 8000) -> pd.DataFrame:
    if len(dataset) <= max_rows:
        return dataset.copy()
    strat = dataset["dose_intensity"].astype(str) + "_" + dataset["retained_horizon"].astype(str)
    sampled_parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(42)
    for _, group in dataset.groupby(strat, dropna=False):
        take = max(1, int(round(len(group) * max_rows / max(len(dataset), 1))))
        if len(group) <= take:
            sampled_parts.append(group)
            continue
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(group.loc[idx])
    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=42)
    return sampled.sort_index().reset_index(drop=True)


def train_dose_response_policy_model(
    dataset: pd.DataFrame,
    *,
    result_dir: Optional[Path] = None,
) -> DoseResponsePolicyModel:
    dataset = _sample_training_dataset(dataset)
    numeric_features, categorical_features = _feature_lists(dataset)
    feature_columns = numeric_features + categorical_features
    X = dataset[feature_columns].copy()
    y = dataset["retained_horizon"].astype(int)
    arms = dataset["dose_intensity"].astype(str)

    propensity_model = _build_classifier(numeric_features, categorical_features)
    propensity_model.fit(X, arms)

    outcome_models: Dict[str, Any] = {}
    arm_summary: Dict[str, Dict[str, float]] = {}
    for arm in DOSE_LEVELS:
        mask = arms == arm
        outcome_models[arm] = _fit_binary_outcome_model(
            X.loc[mask],
            y.loc[mask],
            numeric_features,
            categorical_features,
        )
        avg_cost = float(dataset.loc[mask, "coupon_cost"].mean()) if mask.any() else 0.0
        avg_exposure = float(dataset.loc[mask, "dose_exposure_count"].mean()) if mask.any() else 0.0
        retention_rate = float(dataset.loc[mask, "retained_horizon"].mean()) if mask.any() else 0.0
        avg_revenue = float(dataset.loc[mask, "revenue_post_horizon"].mean()) if mask.any() else 0.0
        arm_summary[arm] = {
            "samples": int(mask.sum()),
            "retention_rate": round(retention_rate, 6),
            "avg_coupon_cost": round(avg_cost, 2),
            "avg_exposure_count": round(avg_exposure, 4),
            "avg_revenue_post_horizon": round(avg_revenue, 2),
        }

    control_prob = _predict_proba(outcome_models[CONTROL_LABEL], X)
    effect_models: Dict[str, Any] = {}
    effect_priors: Dict[str, float] = {}
    for intensity_key in ACTION_INTENSITIES:
        treated_prob = _predict_proba(outcome_models[intensity_key], X)
        direct_effect = np.clip(treated_prob - control_prob, -0.35, 0.75)
        effect_models[intensity_key] = _fit_effect_model(
            X,
            pd.Series(direct_effect, index=X.index),
            numeric_features,
            categorical_features,
        )
        effect_priors[intensity_key] = round(
            float(arm_summary[intensity_key]["retention_rate"] - arm_summary[CONTROL_LABEL]["retention_rate"]),
            6,
        )

    mid_cost = float(dataset.loc[dataset["dose_intensity"] == "mid", "coupon_cost"].mean()) if (dataset["dose_intensity"] == "mid").any() else 0.0
    if mid_cost <= 0:
        mid_cost = float(dataset.loc[dataset["treatment_flag"] == 1, "coupon_cost"].mean()) if (dataset["treatment_flag"] == 1).any() else 1.0
    intensity_cost_multipliers: Dict[str, float] = {}
    for arm in ACTION_INTENSITIES:
        avg_cost = float(arm_summary[arm]["avg_coupon_cost"])
        intensity_cost_multipliers[arm] = round(float(avg_cost / max(mid_cost, 1.0)), 6)

    metadata = {
        "method": "Multi-arm dose-response T-learner over low/mid/high treatment intensity",
        "model_version": DOSE_RESPONSE_MODEL_VERSION,
        "feature_columns": feature_columns,
        "horizon_days": int(DEFAULT_HORIZON_DAYS),
        "training_rows": int(len(dataset)),
        "arm_summary": arm_summary,
        "effect_priors": effect_priors,
        "intensity_cost_multipliers": intensity_cost_multipliers,
    }
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        training_view = dataset[[
            "customer_id",
            "dose_intensity",
            "retained_horizon",
            "revenue_post_horizon",
            "coupon_cost",
            "dose_exposure_count",
        ]].copy()
        training_view["pred_control_prob"] = np.round(control_prob, 6)
        for arm in ACTION_INTENSITIES:
            training_view[f"pred_prob_{arm}"] = np.round(_predict_proba(outcome_models[arm], X), 6)
        training_view.to_csv(result_dir / "dose_response_training_frame.csv", index=False)

    return DoseResponsePolicyModel(
        version=DOSE_RESPONSE_MODEL_VERSION,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        propensity_model=propensity_model,
        outcome_models=outcome_models,
        effect_models=effect_models,
        effect_priors=effect_priors,
        intensity_cost_multipliers=intensity_cost_multipliers,
        arm_summary=arm_summary,
        metadata=metadata,
    )


def fit_and_save_dose_response_policy_model(
    *,
    data_dir: Optional[str | Path] = None,
    model_dir: Optional[str | Path] = None,
    result_dir: Optional[str | Path] = None,
    force_retrain: bool = False,
) -> DoseResponsePolicyModel:
    resolved_data_dir, resolved_model_dir, resolved_result_dir = _resolve_default_paths(
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
    )
    resolved_model_dir.mkdir(parents=True, exist_ok=True)
    resolved_result_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolved_model_dir / "dose_response_policy_model.joblib"
    summary_path = resolved_result_dir / "dose_response_summary.json"

    dependencies = [
        resolved_data_dir / "customer_summary.csv",
        resolved_data_dir / "treatment_assignments.csv",
        resolved_data_dir / "orders.csv",
        resolved_data_dir / "campaign_exposures.csv",
    ]
    latest_dependency = max((path.stat().st_mtime for path in dependencies if path.exists()), default=-1.0)
    model_is_fresh = model_path.exists() and summary_path.exists() and model_path.stat().st_mtime >= latest_dependency

    if (not force_retrain) and model_is_fresh:
        try:
            loaded_model = joblib.load(model_path)
            if isinstance(loaded_model, DoseResponsePolicyModel):
                return loaded_model
        except Exception:
            # 저장된 joblib가 현재 sklearn/runtime 버전과 맞지 않으면 재학습으로 복구한다.
            pass

    dataset = _build_training_dataset(resolved_data_dir)
    model = train_dose_response_policy_model(dataset, result_dir=resolved_result_dir)
    joblib.dump(model, model_path)
    summary_path.write_text(json.dumps(model.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


@lru_cache(maxsize=4)
def load_dose_response_policy_model(
    data_dir: Optional[str | Path] = None,
    model_dir: Optional[str | Path] = None,
    result_dir: Optional[str | Path] = None,
) -> Optional[DoseResponsePolicyModel]:
    resolved_data_dir, resolved_model_dir, resolved_result_dir = _resolve_default_paths(
        data_dir=data_dir,
        model_dir=model_dir,
        result_dir=result_dir,
    )
    model_path = resolved_model_dir / "dose_response_policy_model.joblib"
    summary_path = resolved_result_dir / "dose_response_summary.json"
    if model_path.exists() and summary_path.exists():
        try:
            model = joblib.load(model_path)
            if isinstance(model, DoseResponsePolicyModel):
                return model
        except Exception:
            pass
    try:
        return fit_and_save_dose_response_policy_model(
            data_dir=resolved_data_dir,
            model_dir=resolved_model_dir,
            result_dir=resolved_result_dir,
            force_retrain=False,
        )
    except Exception:
        return None


def load_dose_response_summary(
    result_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    _, _, resolved_result_dir = _resolve_default_paths(result_dir=result_dir)
    path = resolved_result_dir / "dose_response_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
