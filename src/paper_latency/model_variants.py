from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

from src.features.engineering import build_feature_dataset
from src.paper_latency.io_utils import ensure_dir, read_json, write_dataframe, write_json

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


WEAKER_FEATURE_WHITELIST = [
    'customer_age_days',
    'days_since_last_event',
    'recency_days',
    'frequency_30d',
    'frequency_90d',
    'monetary_30d',
    'monetary_90d',
    'visits_14d',
    'visits_prev_14d',
    'visit_change_rate_14d',
    'purchases_14d',
    'purchases_prev_14d',
    'purchase_change_rate_14d',
    'searches_30d',
    'add_to_cart_30d',
    'coupon_open_30d',
    'support_contact_30d',
    'coupon_affinity',
    'price_sensitivity',
    'discount_fatigue_sensitivity',
    'brand_sensitivity',
    'persona',
    'region',
    'device_type',
    'acquisition_channel',
    'current_journey_stage',
]


@dataclass
class VariantArtifacts:
    variant_name: str
    model_path: str
    metrics_path: str
    metrics: dict[str, Any]


@dataclass
class FeatureSnapshot:
    features: pd.DataFrame
    metadata: dict[str, Any]
    csv_path: Path
    metadata_path: Path


class FeatureCache:
    def __init__(self, cache_root: str | Path, *, horizon_days: int = 45) -> None:
        self.cache_root = ensure_dir(cache_root)
        self.horizon_days = int(horizon_days)

    def snapshot_key(self, as_of_date: str | pd.Timestamp) -> str:
        ts = pd.Timestamp(as_of_date).strftime('%Y%m%d')
        return f'asof_{ts}_h{self.horizon_days}'

    def load_or_build(self, data_dir: str | Path, as_of_date: str | pd.Timestamp) -> FeatureSnapshot:
        key = self.snapshot_key(as_of_date)
        csv_path = self.cache_root / f'{key}.csv'
        metadata_path = self.cache_root / f'{key}.json'
        if csv_path.exists() and metadata_path.exists():
            return FeatureSnapshot(
                features=pd.read_csv(csv_path),
                metadata=read_json(metadata_path),
                csv_path=csv_path,
                metadata_path=metadata_path,
            )

        temp_feature_store = ensure_dir(self.cache_root / '_tmp' / key)
        built = build_feature_dataset(
            data_dir=data_dir,
            feature_store_dir=temp_feature_store,
            as_of_date=as_of_date,
            horizon_days=self.horizon_days,
        )
        write_dataframe(csv_path, built.features)
        write_json(metadata_path, built.metadata)
        return FeatureSnapshot(
            features=built.features.copy(),
            metadata=dict(built.metadata),
            csv_path=csv_path,
            metadata_path=metadata_path,
        )


@dataclass
class TrainedVariant:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    training_columns: list[str]

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        X = prepare_design_matrix(features, keep_columns=self.training_columns)
        probs = self.pipeline.predict_proba(X)[:, 1]
        customer_ids = pd.to_numeric(features['customer_id'], errors='coerce').astype('Int64')
        return pd.Series(probs, index=customer_ids.astype(int), name=self.name)


def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    out = series.copy()
    out = out.where(out.notna(), 'unknown')
    out = out.astype('string')
    out = out.fillna('unknown')
    out = out.replace({'nan': 'unknown', '<NA>': 'unknown', 'NaT': 'unknown'})
    return out.astype(str)


def _coerce_categorical_frame(frame: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    out = frame.copy()
    for col in out.columns:
        out[col] = _normalize_categorical_series(out[col])
    return out


def _coerce_numeric_frame(frame: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    out = frame.copy()
    for col in out.columns:
        values = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        out[col] = values.clip(lower=-1_000_000.0, upper=1_000_000.0)
    return out



def _extract_datetime_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    converted: list[str] = []
    for col in out.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist():
        ts = pd.to_datetime(out[col], errors='coerce')
        out[f'{col}_days_from_epoch'] = ((ts - pd.Timestamp('1970-01-01')).dt.total_seconds() / 86400.0)
        out[f'{col}_month'] = ts.dt.month
        out[f'{col}_dayofweek'] = ts.dt.dayofweek
        out.drop(columns=[col], inplace=True)
        converted.append(col)
    return out, converted



def prepare_design_matrix(features_df: pd.DataFrame, keep_columns: Iterable[str] | None = None) -> pd.DataFrame:
    if 'label' in features_df.columns:
        X = features_df.drop(columns=['label', 'customer_id'], errors='ignore').copy()
    else:
        X = features_df.drop(columns=['customer_id'], errors='ignore').copy()

    X, _ = _extract_datetime_features(X)

    if keep_columns is not None:
        keep_set = list(keep_columns)
        missing = [col for col in keep_set if col not in X.columns]
        for column in missing:
            X[column] = np.nan
        X = X.loc[:, keep_set].copy()

    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        elif pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], np.nan).clip(lower=-1_000_000.0, upper=1_000_000.0)
        else:
            X[col] = _normalize_categorical_series(X[col])
    return X



def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    remainder = [col for col in X.columns if col not in cat_cols and col not in num_cols]
    cat_cols.extend(remainder)

    transformers = []
    if num_cols:
        transformers.append((
            'num',
            Pipeline([
                ('coerce_numeric', FunctionTransformer(_coerce_numeric_frame, validate=False)),
                ('imputer', SimpleImputer(strategy='median')),
                ('scale', RobustScaler(with_centering=False, quantile_range=(5.0, 95.0))),
            ]),
            num_cols,
        ))
    if cat_cols:
        transformers.append(
            (
                'cat',
                Pipeline([
                    ('coerce_to_str', FunctionTransformer(_coerce_categorical_frame, validate=False)),
                    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError('No usable columns for churn training.')
    return ColumnTransformer(transformers=transformers, remainder='drop'), num_cols, cat_cols



def _fit_pipeline(X: pd.DataFrame, y: pd.Series, *, variant: str, random_state: int) -> Pipeline:
    preprocessor, _, _ = _build_preprocessor(X)
    if variant == 'weaker':
        estimator = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
    else:
        if XGBClassifier is None:
            raise RuntimeError('xgboost is required for base/stronger variants.')
        params = {
            'n_estimators': 240,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.90,
            'colsample_bytree': 0.85,
            'reg_lambda': 1.0,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_state,
            'n_jobs': 4,
        }
        if variant == 'stronger':
            params.update({
                'n_estimators': 420,
                'max_depth': 6,
                'learning_rate': 0.04,
                'subsample': 0.95,
                'colsample_bytree': 0.95,
                'min_child_weight': 1.0,
            })
        estimator = XGBClassifier(**params)
    return Pipeline([('preprocessor', preprocessor), ('model', estimator)])



def _train_single_variant(X: pd.DataFrame, y: pd.Series, *, variant: str, random_state: int) -> tuple[Pipeline, dict[str, Any]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )
    pipeline = _fit_pipeline(X_train, y_train, variant=variant, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.simplefilter('ignore')
        pipeline.fit(X_train, y_train)
    pred = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, pred))
    ap = float(average_precision_score(y_test, pred))
    metrics = {
        'variant': variant,
        'row_count': int(len(X)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'positive_rate': float(y.mean()),
        'roc_auc': round(auc, 6),
        'average_precision': round(ap, 6),
    }
    return pipeline, metrics



def _build_training_panel(data_dir: str | Path, cache: FeatureCache, training_dates: list[pd.Timestamp]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for as_of_date in training_dates:
        snap = cache.load_or_build(data_dir=data_dir, as_of_date=as_of_date)
        frame = snap.features.copy()
        if 'label' not in frame.columns:
            skipped.append(f"{pd.Timestamp(as_of_date).date()}:missing-label")
            continue
        positive_rate = float(pd.to_numeric(frame['label'], errors='coerce').fillna(0).mean())
        if positive_rate <= 0.0 or positive_rate >= 1.0:
            skipped.append(f"{pd.Timestamp(as_of_date).date()}:degenerate-{positive_rate:.4f}")
            continue
        frame['training_as_of_date'] = str(pd.Timestamp(as_of_date).date())
        frames.append(frame)
    if not frames:
        raise ValueError(f'No usable training frames were created. skipped={skipped}')
    return pd.concat(frames, ignore_index=True)



def train_variants_for_seed(
    *,
    seed: int,
    data_dir: str | Path,
    cache_dir: str | Path,
    model_dir: str | Path,
    result_dir: str | Path,
    training_dates: list[pd.Timestamp],
    random_state: int,
) -> dict[str, VariantArtifacts]:
    cache = FeatureCache(cache_dir, horizon_days=45)
    training_panel = _build_training_panel(data_dir=data_dir, cache=cache, training_dates=training_dates)
    if 'label' not in training_panel.columns:
        raise ValueError('Training panel must include label.')

    y = training_panel['label'].astype(int)
    full_X = prepare_design_matrix(training_panel)
    weaker_columns = [col for col in WEAKER_FEATURE_WHITELIST if col in full_X.columns]
    if not weaker_columns:
        raise ValueError('Weaker feature whitelist produced zero usable columns.')

    ensure_dir(model_dir)
    ensure_dir(result_dir)
    write_dataframe(Path(result_dir) / f'seed_{seed}_training_panel.csv', training_panel)

    artifacts: dict[str, VariantArtifacts] = {}
    for variant_name, X in {
        'base': full_X,
        'stronger': full_X,
        'weaker': full_X.loc[:, weaker_columns].copy(),
    }.items():
        pipeline, metrics = _train_single_variant(
            X,
            y,
            variant=variant_name,
            random_state=random_state + int(seed),
        )
        payload = {
            'variant_name': variant_name,
            'training_columns': list(X.columns),
            'metrics': metrics,
        }
        model_path = Path(model_dir) / f'seed_{seed}_{variant_name}_model.joblib'
        metrics_path = Path(result_dir) / f'seed_{seed}_{variant_name}_metrics.json'
        joblib.dump({'pipeline': pipeline, **payload}, model_path)
        write_json(metrics_path, payload)
        artifacts[variant_name] = VariantArtifacts(
            variant_name=variant_name,
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            metrics=payload,
        )
    return artifacts



def load_trained_variant(model_path: str | Path) -> TrainedVariant:
    blob = joblib.load(model_path)
    return TrainedVariant(
        name=str(blob['variant_name']),
        pipeline=blob['pipeline'],
        metrics=dict(blob.get('metrics', {})),
        training_columns=list(blob.get('training_columns', [])),
    )
