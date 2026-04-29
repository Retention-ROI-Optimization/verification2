from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SEEDS: tuple[int, ...] = (41, 42, 43, 44, 45)
DEFAULT_SCENARIO_FAMILIES: tuple[str, ...] = (
    'complaint-heavy',
    'promotion-heavy',
    'dormancy-heavy',
    'seasonal-shift',
)
DEFAULT_LATENCIES: tuple[int, ...] = (0, 1, 3, 7)
DEFAULT_BUDGETS: tuple[int, ...] = (2_640_000, 7_250_000, 11_530_000)
DEFAULT_RETRAIN_LATENCY_FOR_MODEL_COMPARISON = 3


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    raw_grid_dir: Path
    cache_dir: Path
    model_dir: Path
    result_dir: Path
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    scenario_families: tuple[str, ...] = DEFAULT_SCENARIO_FAMILIES
    latencies: tuple[int, ...] = DEFAULT_LATENCIES
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    burn_in_weeks: int = 12
    horizon_days: int = 45
    training_landmarks: int = 12
    bootstrap_iterations: int = 1000
    random_state: int = 42
    decision_week_limit: int | None = None
    stronger_vs_weaker_latency_days: int = DEFAULT_RETRAIN_LATENCY_FOR_MODEL_COMPARISON
    partial_reopt_score_delta: float = 0.10
    partial_reopt_high_risk_threshold: float = 0.80
    partial_reopt_top_share: float = 0.15
    use_learned_dose_response: bool = False
    # ── Conformal Risk Control ──
    conformal_alpha_grid: tuple[float, ...] = (0.05, 0.10, 0.20)
    conformal_min_cal_size: int = 200
    ensemble_size: int = 5

    @classmethod
    def from_root(cls, project_root: str | Path, **overrides) -> 'ExperimentConfig':
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            raw_grid_dir=root / 'artifacts' / 'raw_grid',
            cache_dir=root / 'artifacts' / 'feature_cache',
            model_dir=root / 'artifacts' / 'models',
            result_dir=root / 'artifacts' / 'results',
            **overrides,
        )


def parse_int_list(raw: str | Iterable[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(x) for x in default)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(',') if item.strip()]
        return tuple(int(x) for x in values)
    return tuple(int(x) for x in raw)


def parse_str_list(raw: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(str(x) for x in default)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(',') if item.strip()]
        return tuple(values)
    return tuple(str(x) for x in raw)


def parse_float_list(raw: str | Iterable[float] | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(float(x) for x in default)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(',') if item.strip()]
        return tuple(float(x) for x in values)
    return tuple(float(x) for x in raw)
