from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paper_latency.config import ExperimentConfig
from src.paper_latency.engine import (
    compute_policy_comparison_metrics,
    matched_reoptimization_policy,
    partial_reoptimization,
    run_policy_selection,
    select_random_reopt_ids,
    select_top_risk_reopt_ids,
    select_top_value_reopt_ids,
)
from src.paper_latency.io_utils import ensure_dir, read_json, seed_dir, write_dataframe, write_json
from src.paper_latency.model_variants import FeatureCache, load_trained_variant, train_variants_for_seed
from src.paper_latency.scenario_family import apply_scenario_family
from src.simulator.config import DEFAULT_CONFIG
from src.simulator.pipeline import run_simulation


SEED_MODEL_INDEX = {
    'base': 'base',
    'stronger': 'stronger',
    'weaker': 'weaker',
}


class PaperExperimentError(RuntimeError):
    pass



def _seed_raw_dir(config: ExperimentConfig, seed: int) -> Path:
    return seed_dir(config.raw_grid_dir, seed)



def prepare_simulation_grid(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    produced: dict[str, Any] = {'prepared_seeds': [], 'raw_grid_dir': str(config.raw_grid_dir)}
    for seed in config.seeds:
        data_dir = _seed_raw_dir(config, seed)
        required = [
            data_dir / 'customers.csv',
            data_dir / 'events.csv',
            data_dir / 'orders.csv',
            data_dir / 'state_snapshots.csv',
            data_dir / 'campaign_exposures.csv',
            data_dir / 'treatment_assignments.csv',
            data_dir / 'customer_summary.csv',
            data_dir / 'cohort_retention.csv',
        ]
        if not force and all(path.exists() for path in required):
            produced['prepared_seeds'].append({'seed': seed, 'data_dir': str(data_dir), 'status': 'reused'})
            continue
        ensure_dir(data_dir)
        sim_config = DEFAULT_CONFIG.with_seed(int(seed))
        run_simulation(config=sim_config, export=True, output_dir=str(data_dir), file_format='csv')
        produced['prepared_seeds'].append({'seed': seed, 'data_dir': str(data_dir), 'status': 'generated'})
    write_json(config.result_dir / 'prepare_simulation_grid.json', produced)
    return produced



def _decision_schedule(data_dir: str | Path, *, burn_in_weeks: int, limit: int | None) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    snapshots = pd.read_csv(Path(data_dir) / 'state_snapshots.csv', parse_dates=['snapshot_date'])
    schedule = sorted(pd.to_datetime(snapshots['snapshot_date']).dt.floor('D').drop_duplicates().tolist())
    if len(schedule) <= burn_in_weeks:
        raise PaperExperimentError(f'Not enough snapshot weeks. Found {len(schedule)}, need more than burn-in {burn_in_weeks}.')
    training_dates = [pd.Timestamp(ts) for ts in schedule[:burn_in_weeks]]
    decision_dates = [pd.Timestamp(ts) for ts in schedule[burn_in_weeks:]]
    if limit is not None:
        decision_dates = decision_dates[: int(limit)]
    return training_dates, decision_dates



def train_all_seed_variants(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    prepare_simulation_grid(config, force=False)
    payload: dict[str, Any] = {'trained_seeds': []}
    for seed in config.seeds:
        data_dir = _seed_raw_dir(config, seed)
        training_dates, _ = _decision_schedule(data_dir, burn_in_weeks=config.burn_in_weeks, limit=config.decision_week_limit)
        model_dir = ensure_dir(config.model_dir / f'seed_{seed}')
        result_dir = ensure_dir(config.result_dir / 'training' / f'seed_{seed}')
        expected_paths = [model_dir / f'seed_{seed}_{name}_model.joblib' for name in ['base', 'stronger', 'weaker']]
        if (not force) and all(path.exists() for path in expected_paths):
            payload['trained_seeds'].append({'seed': seed, 'status': 'reused', 'model_dir': str(model_dir)})
            continue
        artifacts = train_variants_for_seed(
            seed=seed,
            data_dir=data_dir,
            cache_dir=config.cache_dir / f'seed_{seed}',
            model_dir=model_dir,
            result_dir=result_dir,
            training_dates=training_dates[-config.training_landmarks :],
            random_state=config.random_state,
        )
        payload['trained_seeds'].append({'seed': seed, 'status': 'trained', 'artifacts': {name: asdict(artifact) for name, artifact in artifacts.items()}})
    write_json(config.result_dir / 'train_variants_summary.json', payload)
    return payload



def _load_seed_models(config: ExperimentConfig, seed: int) -> dict[str, Any]:
    seed_model_dir = config.model_dir / f'seed_{seed}'
    return {
        name: load_trained_variant(seed_model_dir / f'seed_{seed}_{name}_model.joblib')
        for name in ['base', 'stronger', 'weaker']
    }



def _fresh_and_stale_snapshots(
    *,
    cache: FeatureCache,
    data_dir: str | Path,
    decision_date: pd.Timestamp,
    latencies: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    snapshots: dict[int, pd.DataFrame] = {}
    for latency in latencies:
        as_of = decision_date - pd.Timedelta(days=int(latency))
        snap = cache.load_or_build(data_dir=data_dir, as_of_date=as_of)
        snapshots[int(latency)] = snap.features.copy()
    return snapshots



def _score_variant(variant, features: pd.DataFrame) -> pd.Series:
    scores = variant.predict_proba(features)
    scores.name = variant.name
    return scores



def _bootstrap_interval(values: np.ndarray, *, iterations: int, rng: np.random.Generator) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    if values.size == 1:
        single = float(values[0])
        return single, single, single
    means = []
    for _ in range(int(iterations)):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    arr = np.asarray(means, dtype=float)
    return float(np.mean(values)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))



def _summarize_metrics(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric_cols: list[str],
    bootstrap_iterations: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for idx, (group_key, group) in enumerate(grouped):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, Any] = {column: value for column, value in zip(group_cols, group_key)}
        row['block_count'] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors='coerce').dropna().to_numpy(dtype=float)
            rng = np.random.default_rng(random_state + idx * 97 + len(metric))
            mean, low, high = _bootstrap_interval(values, iterations=bootstrap_iterations, rng=rng)
            row[f'{metric}_mean'] = round(mean, 6)
            row[f'{metric}_ci_low'] = round(low, 6)
            row[f'{metric}_ci_high'] = round(high, 6)
        rows.append(row)
    return pd.DataFrame(rows)





def _safe_recovery(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _add_selection_metrics(
    row: dict[str, Any],
    *,
    prefix: str,
    fresh_policy,
    stale_metrics: dict[str, Any],
    selection,
    meta: dict[str, Any],
    latency: int,
) -> None:
    metrics = compute_policy_comparison_metrics(
        fresh_selection=fresh_policy,
        candidate_selection=selection,
        latency_days=latency,
    )
    row[f'{prefix}_policy_value'] = round(metrics['policy_value'], 6)
    row[f'{prefix}_stale_regret'] = round(metrics['stale_regret'], 6)
    row[f'{prefix}_relative_loss'] = round(metrics['relative_loss'], 6)
    row[f'{prefix}_target_overlap'] = round(metrics['target_overlap'], 6)
    row[f'{prefix}_missed_at_risk'] = round(metrics['missed_at_risk'], 6)
    row[f'{prefix}_window_miss_rate'] = round(metrics['window_miss_rate'], 6)
    row[f'{prefix}_selected_customers'] = int(metrics['selected_customers'])
    row[f'{prefix}_full_refresh_value_ratio'] = round(
        _full_refresh_ratio(metrics['policy_value'], fresh_policy.summary['policy_value']),
        6,
    )
    row[f'{prefix}_optimization_call_ratio'] = float(meta.get('optimization_call_ratio', 0.0))
    row[f'{prefix}_target_overlap_recovery'] = round(
        _safe_recovery(metrics['target_overlap'] - stale_metrics['target_overlap'], 1.0 - stale_metrics['target_overlap']),
        6,
    )
    row[f'{prefix}_missed_at_risk_recovery'] = round(
        _safe_recovery(stale_metrics['missed_at_risk'] - metrics['missed_at_risk'], stale_metrics['missed_at_risk']),
        6,
    )
    row[f'{prefix}_window_miss_rate_recovery'] = round(
        _safe_recovery(stale_metrics['window_miss_rate'] - metrics['window_miss_rate'], stale_metrics['window_miss_rate']),
        6,
    )


def _summarize_block_metrics(block_metrics: pd.DataFrame, *, bootstrap_iterations: int, random_state: int) -> pd.DataFrame:
    metric_cols = [
        'policy_value',
        'stale_regret',
        'relative_loss',
        'target_overlap',
        'missed_at_risk',
        'window_miss_rate',
        'selected_customers',
        'partial_reopt_optimization_call_ratio',
        'partial_reopt_regret_recovery_ratio',
        'partial_reopt_full_refresh_value_ratio',
        'partial_reopt_target_overlap',
        'partial_reopt_missed_at_risk',
        'partial_reopt_window_miss_rate',
        'partial_reopt_relative_loss',
        'partial_reopt_target_overlap_recovery',
        'partial_reopt_missed_at_risk_recovery',
        'partial_reopt_window_miss_rate_recovery',
        'random_refresh_full_refresh_value_ratio',
        'random_refresh_target_overlap',
        'random_refresh_missed_at_risk',
        'random_refresh_window_miss_rate',
        'random_refresh_relative_loss',
        'random_refresh_target_overlap_recovery',
        'random_refresh_missed_at_risk_recovery',
        'random_refresh_window_miss_rate_recovery',
        'top_risk_refresh_full_refresh_value_ratio',
        'top_risk_refresh_target_overlap',
        'top_risk_refresh_missed_at_risk',
        'top_risk_refresh_window_miss_rate',
        'top_risk_refresh_relative_loss',
        'top_risk_refresh_target_overlap_recovery',
        'top_risk_refresh_missed_at_risk_recovery',
        'top_risk_refresh_window_miss_rate_recovery',
        'top_value_refresh_full_refresh_value_ratio',
        'top_value_refresh_target_overlap',
        'top_value_refresh_missed_at_risk',
        'top_value_refresh_window_miss_rate',
        'top_value_refresh_relative_loss',
        'top_value_refresh_target_overlap_recovery',
        'top_value_refresh_missed_at_risk_recovery',
        'top_value_refresh_window_miss_rate_recovery',
    ]
    existing_metric_cols = [col for col in metric_cols if col in block_metrics.columns]
    return _summarize_metrics(
        block_metrics,
        group_cols=['scenario_family', 'budget', 'policy_kind', 'latency_days'],
        metric_cols=existing_metric_cols,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )



def _full_refresh_ratio(partial_value: float, fresh_value: float) -> float:
    if abs(fresh_value) < 1e-9:
        return 0.0
    return partial_value / fresh_value



def run_rolling_latency_evaluation(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    prepare_simulation_grid(config, force=False)
    train_all_seed_variants(config, force=False)
    result_root = ensure_dir(config.result_dir / 'paper_latency')
    block_metrics_path = result_root / 'block_level_metrics.csv'
    summary_path = result_root / 'summary_metrics.csv'
    manifest_path = result_root / 'manifest.json'
    if (not force) and block_metrics_path.exists() and summary_path.exists() and manifest_path.exists():
        return read_json(manifest_path)

    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        data_dir = _seed_raw_dir(config, seed)
        models = _load_seed_models(config, seed)
        cache = FeatureCache(config.cache_dir / f'seed_{seed}', horizon_days=config.horizon_days)
        _, decision_dates = _decision_schedule(data_dir, burn_in_weeks=config.burn_in_weeks, limit=config.decision_week_limit)

        for decision_date in decision_dates:
            eval_latencies = tuple(sorted(set(config.latencies)))
            required_latencies = tuple(sorted({0, *eval_latencies}))
            raw_feature_snapshots = _fresh_and_stale_snapshots(
                cache=cache,
                data_dir=data_dir,
                decision_date=decision_date,
                latencies=required_latencies,
            )

            for family in config.scenario_families:
                fresh_features = apply_scenario_family(raw_feature_snapshots[0], family, decision_date).features
                family_snapshots = {
                    latency: apply_scenario_family(raw_feature_snapshots[latency], family, decision_date).features
                    for latency in eval_latencies
                }
                fresh_base_scores = _score_variant(models['base'], fresh_features)
                fresh_policy_by_budget = {
                    budget: run_policy_selection(
                        fresh_features=fresh_features,
                        churn_scores=fresh_base_scores,
                        budget=budget,
                        scenario_family=family,
                        decision_date=decision_date,
                        use_learned_dose_response=config.use_learned_dose_response,
                    )
                    for budget in config.budgets
                }
                weaker_fresh_scores = _score_variant(models['weaker'], fresh_features)

                for latency in config.latencies:
                    stale_features = family_snapshots[latency]
                    base_scores = _score_variant(models['base'], stale_features)
                    stronger_scores = _score_variant(models['stronger'], stale_features)

                    for budget in config.budgets:
                        fresh_policy = fresh_policy_by_budget[budget]
                        stale_policy = run_policy_selection(
                            fresh_features=fresh_features,
                            churn_scores=base_scores,
                            budget=budget,
                            scenario_family=family,
                            decision_date=decision_date,
                            use_learned_dose_response=config.use_learned_dose_response,
                        )
                        stale_metrics = compute_policy_comparison_metrics(
                            fresh_selection=fresh_policy,
                            candidate_selection=stale_policy,
                            latency_days=latency,
                        )
                        row = {
                            'seed': int(seed),
                            'scenario_family': family,
                            'decision_date': str(pd.Timestamp(decision_date).date()),
                            'budget': int(budget),
                            'policy_kind': 'base-stale',
                            'latency_days': int(latency),
                            **stale_metrics,
                        }

                        partial_policy, partial_meta = partial_reoptimization(
                            stale_scores=base_scores,
                            fresh_scores=fresh_base_scores,
                            fresh_features=fresh_features,
                            stale_selection=stale_policy,
                            budget=budget,
                            scenario_family=family,
                            decision_date=decision_date,
                            score_delta_threshold=config.partial_reopt_score_delta,
                            high_risk_threshold=config.partial_reopt_high_risk_threshold,
                            top_share=config.partial_reopt_top_share,
                            use_learned_dose_response=config.use_learned_dose_response,
                        )
                        partial_metrics = compute_policy_comparison_metrics(
                            fresh_selection=fresh_policy,
                            candidate_selection=partial_policy,
                            latency_days=latency,
                        )
                        row['partial_reopt_policy_value'] = round(partial_metrics['policy_value'], 6)
                        row['partial_reopt_stale_regret'] = round(partial_metrics['stale_regret'], 6)
                        row['partial_reopt_relative_loss'] = round(partial_metrics['relative_loss'], 6)
                        row['partial_reopt_target_overlap'] = round(partial_metrics['target_overlap'], 6)
                        row['partial_reopt_missed_at_risk'] = round(partial_metrics['missed_at_risk'], 6)
                        row['partial_reopt_window_miss_rate'] = round(partial_metrics['window_miss_rate'], 6)
                        row['partial_reopt_selected_customers'] = int(partial_metrics['selected_customers'])
                        row['partial_reopt_regret_recovery_ratio'] = round(
                            (stale_metrics['stale_regret'] - partial_metrics['stale_regret']) / max(abs(stale_metrics['stale_regret']), 1.0),
                            6,
                        )
                        row['partial_reopt_full_refresh_value_ratio'] = round(
                            _full_refresh_ratio(partial_metrics['policy_value'], fresh_policy.summary['policy_value']),
                            6,
                        )
                        row['partial_reopt_optimization_call_ratio'] = float(partial_meta['optimization_call_ratio'])
                        row['partial_reopt_target_overlap_recovery'] = round(
                            _safe_recovery(partial_metrics['target_overlap'] - stale_metrics['target_overlap'], 1.0 - stale_metrics['target_overlap']),
                            6,
                        )
                        row['partial_reopt_missed_at_risk_recovery'] = round(
                            _safe_recovery(stale_metrics['missed_at_risk'] - partial_metrics['missed_at_risk'], stale_metrics['missed_at_risk']),
                            6,
                        )
                        row['partial_reopt_window_miss_rate_recovery'] = round(
                            _safe_recovery(stale_metrics['window_miss_rate'] - partial_metrics['window_miss_rate'], stale_metrics['window_miss_rate']),
                            6,
                        )

                        matched_k = int(partial_meta.get('reoptimized_customers', 0))
                        if matched_k > 0:
                            family_code = sum(ord(ch) for ch in family)
                            rng = np.random.default_rng(
                                int(config.random_state + seed * 100003 + pd.Timestamp(decision_date).toordinal() * 37 + int(budget) * 7 + int(latency) * 997 + family_code)
                            )
                            random_ids = select_random_reopt_ids(fresh_base_scores.index, k=matched_k, rng=rng)
                            random_policy, random_meta = matched_reoptimization_policy(
                                stale_scores=base_scores,
                                fresh_scores=fresh_base_scores,
                                fresh_features=fresh_features,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                reopt_ids=random_ids,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            _add_selection_metrics(
                                row,
                                prefix='random_refresh',
                                fresh_policy=fresh_policy,
                                stale_metrics=stale_metrics,
                                selection=random_policy,
                                meta=random_meta,
                                latency=latency,
                            )

                            top_risk_ids = select_top_risk_reopt_ids(base_scores, k=matched_k)
                            top_risk_policy, top_risk_meta = matched_reoptimization_policy(
                                stale_scores=base_scores,
                                fresh_scores=fresh_base_scores,
                                fresh_features=fresh_features,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                reopt_ids=top_risk_ids,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            _add_selection_metrics(
                                row,
                                prefix='top_risk_refresh',
                                fresh_policy=fresh_policy,
                                stale_metrics=stale_metrics,
                                selection=top_risk_policy,
                                meta=top_risk_meta,
                                latency=latency,
                            )

                            top_value_ids = select_top_value_reopt_ids(stale_policy, k=matched_k, fallback_scores=base_scores)
                            top_value_policy, top_value_meta = matched_reoptimization_policy(
                                stale_scores=base_scores,
                                fresh_scores=fresh_base_scores,
                                fresh_features=fresh_features,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                reopt_ids=top_value_ids,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            _add_selection_metrics(
                                row,
                                prefix='top_value_refresh',
                                fresh_policy=fresh_policy,
                                stale_metrics=stale_metrics,
                                selection=top_value_policy,
                                meta=top_value_meta,
                                latency=latency,
                            )
                        rows.append(row)

                        if latency == config.stronger_vs_weaker_latency_days:
                            stronger_policy = run_policy_selection(
                                fresh_features=fresh_features,
                                churn_scores=stronger_scores,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            stronger_metrics = compute_policy_comparison_metrics(
                                fresh_selection=fresh_policy,
                                candidate_selection=stronger_policy,
                                latency_days=latency,
                            )
                            rows.append(
                                {
                                    'seed': int(seed),
                                    'scenario_family': family,
                                    'decision_date': str(pd.Timestamp(decision_date).date()),
                                    'budget': int(budget),
                                    'policy_kind': 'stronger-but-stale',
                                    'latency_days': int(latency),
                                    **stronger_metrics,
                                }
                            )
                            weaker_policy = run_policy_selection(
                                fresh_features=fresh_features,
                                churn_scores=weaker_fresh_scores,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            weaker_metrics = compute_policy_comparison_metrics(
                                fresh_selection=fresh_policy,
                                candidate_selection=weaker_policy,
                                latency_days=0,
                            )
                            rows.append(
                                {
                                    'seed': int(seed),
                                    'scenario_family': family,
                                    'decision_date': str(pd.Timestamp(decision_date).date()),
                                    'budget': int(budget),
                                    'policy_kind': 'weaker-but-fresh',
                                    'latency_days': 0,
                                    **weaker_metrics,
                                }
                            )

                # Explicitly export the fresh full-refresh policy once per budget/week/family.
                for budget, fresh_policy in fresh_policy_by_budget.items():
                    rows.append(
                        {
                            'seed': int(seed),
                            'scenario_family': family,
                            'decision_date': str(pd.Timestamp(decision_date).date()),
                            'budget': int(budget),
                            'policy_kind': 'full-refresh',
                            'latency_days': 0,
                            'policy_value': round(float(fresh_policy.summary['policy_value']), 6),
                            'fresh_policy_value': round(float(fresh_policy.summary['policy_value']), 6),
                            'stale_regret': 0.0,
                            'relative_loss': 0.0,
                            'target_overlap': 1.0,
                            'missed_at_risk': 0.0,
                            'window_miss_rate': 0.0,
                            'selected_customers': int(len(fresh_policy.selected)),
                            'partial_reopt_policy_value': round(float(fresh_policy.summary['policy_value']), 6),
                            'partial_reopt_stale_regret': 0.0,
                            'partial_reopt_regret_recovery_ratio': 1.0,
                            'partial_reopt_full_refresh_value_ratio': 1.0,
                            'partial_reopt_optimization_call_ratio': 1.0,
                            'partial_reopt_target_overlap': 1.0,
                            'partial_reopt_missed_at_risk': 0.0,
                            'partial_reopt_window_miss_rate': 0.0,
                            'partial_reopt_relative_loss': 0.0,
                            'partial_reopt_target_overlap_recovery': 1.0,
                            'partial_reopt_missed_at_risk_recovery': 1.0,
                            'partial_reopt_window_miss_rate_recovery': 1.0,
                            'random_refresh_full_refresh_value_ratio': 1.0,
                            'random_refresh_target_overlap': 1.0,
                            'random_refresh_missed_at_risk': 0.0,
                            'random_refresh_window_miss_rate': 0.0,
                            'random_refresh_relative_loss': 0.0,
                            'random_refresh_target_overlap_recovery': 1.0,
                            'random_refresh_missed_at_risk_recovery': 1.0,
                            'random_refresh_window_miss_rate_recovery': 1.0,
                            'top_risk_refresh_full_refresh_value_ratio': 1.0,
                            'top_risk_refresh_target_overlap': 1.0,
                            'top_risk_refresh_missed_at_risk': 0.0,
                            'top_risk_refresh_window_miss_rate': 0.0,
                            'top_risk_refresh_relative_loss': 0.0,
                            'top_risk_refresh_target_overlap_recovery': 1.0,
                            'top_risk_refresh_missed_at_risk_recovery': 1.0,
                            'top_risk_refresh_window_miss_rate_recovery': 1.0,
                            'top_value_refresh_full_refresh_value_ratio': 1.0,
                            'top_value_refresh_target_overlap': 1.0,
                            'top_value_refresh_missed_at_risk': 0.0,
                            'top_value_refresh_window_miss_rate': 0.0,
                            'top_value_refresh_relative_loss': 0.0,
                            'top_value_refresh_target_overlap_recovery': 1.0,
                            'top_value_refresh_missed_at_risk_recovery': 1.0,
                            'top_value_refresh_window_miss_rate_recovery': 1.0,
                        }
                    )

    block_metrics = pd.DataFrame(rows)
    if block_metrics.empty:
        raise PaperExperimentError('No block-level metrics were generated.')

    summary = _summarize_block_metrics(
        block_metrics,
        bootstrap_iterations=config.bootstrap_iterations,
        random_state=config.random_state,
    )

    write_dataframe(block_metrics_path, block_metrics)
    write_dataframe(summary_path, summary)
    manifest = {
        'block_metrics_path': str(block_metrics_path),
        'summary_path': str(summary_path),
        'rows': int(len(block_metrics)),
        'summary_rows': int(len(summary)),
        'config': {
            'seeds': list(config.seeds),
            'scenario_families': list(config.scenario_families),
            'latencies': list(config.latencies),
            'budgets': list(config.budgets),
            'burn_in_weeks': int(config.burn_in_weeks),
            'decision_week_limit': config.decision_week_limit,
            'bootstrap_iterations': int(config.bootstrap_iterations),
        },
    }
    write_json(manifest_path, manifest)
    return manifest





def _theta_grid_slug(theta_grid: tuple[float, ...]) -> str:
    def _fmt(value: float) -> str:
        return f'{float(value):.3f}'.rstrip('0').rstrip('.').replace('-', 'm').replace('.', 'p')

    return '__'.join(_fmt(value) for value in theta_grid)



def run_theta_sensitivity(
    config: ExperimentConfig,
    *,
    theta_grid: tuple[float, ...],
    force: bool = False,
) -> dict[str, Any]:
    if not theta_grid:
        raise PaperExperimentError('theta_grid must not be empty.')

    prepare_simulation_grid(config, force=False)
    train_all_seed_variants(config, force=False)

    theta_grid = tuple(sorted({round(float(theta), 6) for theta in theta_grid}))
    slug = _theta_grid_slug(theta_grid)
    result_root = ensure_dir(config.result_dir / 'paper_latency' / 'theta_sensitivity' / slug)
    block_metrics_path = result_root / 'theta_block_level_metrics.csv'
    latency_summary_path = result_root / 'theta_summary_by_latency.csv'
    overall_summary_path = result_root / 'theta_summary_overall.csv'
    manifest_path = result_root / 'manifest.json'
    if (not force) and block_metrics_path.exists() and latency_summary_path.exists() and overall_summary_path.exists() and manifest_path.exists():
        return read_json(manifest_path)

    rows: list[dict[str, Any]] = []
    metric_cols = [
        'base_policy_value',
        'base_stale_regret',
        'base_relative_loss',
        'base_target_overlap',
        'base_missed_at_risk',
        'base_window_miss_rate',
        'partial_reopt_policy_value',
        'partial_reopt_stale_regret',
        'partial_reopt_regret_recovery_ratio',
        'partial_reopt_full_refresh_value_ratio',
        'partial_reopt_optimization_call_ratio',
    ]

    for seed in config.seeds:
        data_dir = _seed_raw_dir(config, seed)
        models = _load_seed_models(config, seed)
        cache = FeatureCache(config.cache_dir / f'seed_{seed}', horizon_days=config.horizon_days)
        _, decision_dates = _decision_schedule(data_dir, burn_in_weeks=config.burn_in_weeks, limit=config.decision_week_limit)

        for decision_date in decision_dates:
            eval_latencies = tuple(sorted(set(config.latencies)))
            required_latencies = tuple(sorted({0, *eval_latencies}))
            raw_feature_snapshots = _fresh_and_stale_snapshots(
                cache=cache,
                data_dir=data_dir,
                decision_date=decision_date,
                latencies=required_latencies,
            )

            for family in config.scenario_families:
                fresh_features = apply_scenario_family(raw_feature_snapshots[0], family, decision_date).features
                family_snapshots = {
                    latency: apply_scenario_family(raw_feature_snapshots[latency], family, decision_date).features
                    for latency in eval_latencies
                }
                fresh_base_scores = _score_variant(models['base'], fresh_features)
                fresh_policy_by_budget = {
                    budget: run_policy_selection(
                        fresh_features=fresh_features,
                        churn_scores=fresh_base_scores,
                        budget=budget,
                        scenario_family=family,
                        decision_date=decision_date,
                        use_learned_dose_response=config.use_learned_dose_response,
                    )
                    for budget in config.budgets
                }

                for latency in config.latencies:
                    stale_features = family_snapshots[latency]
                    base_scores = _score_variant(models['base'], stale_features)

                    for budget in config.budgets:
                        fresh_policy = fresh_policy_by_budget[budget]
                        stale_policy = run_policy_selection(
                            fresh_features=fresh_features,
                            churn_scores=base_scores,
                            budget=budget,
                            scenario_family=family,
                            decision_date=decision_date,
                            use_learned_dose_response=config.use_learned_dose_response,
                        )
                        stale_metrics = compute_policy_comparison_metrics(
                            fresh_selection=fresh_policy,
                            candidate_selection=stale_policy,
                            latency_days=latency,
                        )

                        for theta in theta_grid:
                            partial_policy, partial_meta = partial_reoptimization(
                                stale_scores=base_scores,
                                fresh_scores=fresh_base_scores,
                                fresh_features=fresh_features,
                                stale_selection=stale_policy,
                                budget=budget,
                                scenario_family=family,
                                decision_date=decision_date,
                                score_delta_threshold=float(theta),
                                high_risk_threshold=config.partial_reopt_high_risk_threshold,
                                top_share=config.partial_reopt_top_share,
                                use_learned_dose_response=config.use_learned_dose_response,
                            )
                            partial_metrics = compute_policy_comparison_metrics(
                                fresh_selection=fresh_policy,
                                candidate_selection=partial_policy,
                                latency_days=latency,
                            )
                            rows.append(
                                {
                                    'seed': int(seed),
                                    'scenario_family': family,
                                    'decision_date': str(pd.Timestamp(decision_date).date()),
                                    'budget': int(budget),
                                    'latency_days': int(latency),
                                    'theta': float(theta),
                                    'high_risk_threshold': float(config.partial_reopt_high_risk_threshold),
                                    'top_share': float(config.partial_reopt_top_share),
                                    'base_policy_value': round(stale_metrics['policy_value'], 6),
                                    'base_stale_regret': round(stale_metrics['stale_regret'], 6),
                                    'base_relative_loss': round(stale_metrics['relative_loss'], 6),
                                    'base_target_overlap': round(stale_metrics['target_overlap'], 6),
                                    'base_missed_at_risk': round(stale_metrics['missed_at_risk'], 6),
                                    'base_window_miss_rate': round(stale_metrics['window_miss_rate'], 6),
                                    'partial_reopt_policy_value': round(partial_metrics['policy_value'], 6),
                                    'partial_reopt_stale_regret': round(partial_metrics['stale_regret'], 6),
                                    'partial_reopt_regret_recovery_ratio': round(
                                        (stale_metrics['stale_regret'] - partial_metrics['stale_regret']) / max(abs(stale_metrics['stale_regret']), 1.0),
                                        6,
                                    ),
                                    'partial_reopt_full_refresh_value_ratio': round(
                                        _full_refresh_ratio(partial_metrics['policy_value'], fresh_policy.summary['policy_value']),
                                        6,
                                    ),
                                    'partial_reopt_optimization_call_ratio': float(partial_meta['optimization_call_ratio']),
                                }
                            )

    theta_block_metrics = pd.DataFrame(rows)
    if theta_block_metrics.empty:
        raise PaperExperimentError('No theta sensitivity metrics were generated.')

    latency_summary = _summarize_metrics(
        theta_block_metrics,
        group_cols=['theta', 'latency_days'],
        metric_cols=metric_cols,
        bootstrap_iterations=config.bootstrap_iterations,
        random_state=config.random_state,
    )
    overall_summary = _summarize_metrics(
        theta_block_metrics,
        group_cols=['theta'],
        metric_cols=metric_cols,
        bootstrap_iterations=config.bootstrap_iterations,
        random_state=config.random_state + 701,
    )

    write_dataframe(block_metrics_path, theta_block_metrics)
    write_dataframe(latency_summary_path, latency_summary)
    write_dataframe(overall_summary_path, overall_summary)

    manifest = {
        'theta_block_metrics_path': str(block_metrics_path),
        'theta_summary_by_latency_path': str(latency_summary_path),
        'theta_summary_overall_path': str(overall_summary_path),
        'rows': int(len(theta_block_metrics)),
        'latency_summary_rows': int(len(latency_summary)),
        'overall_summary_rows': int(len(overall_summary)),
        'config': {
            'seeds': list(config.seeds),
            'scenario_families': list(config.scenario_families),
            'latencies': list(config.latencies),
            'budgets': list(config.budgets),
            'burn_in_weeks': int(config.burn_in_weeks),
            'decision_week_limit': config.decision_week_limit,
            'bootstrap_iterations': int(config.bootstrap_iterations),
            'theta_grid': list(theta_grid),
            'partial_reopt_high_risk_threshold': float(config.partial_reopt_high_risk_threshold),
            'partial_reopt_top_share': float(config.partial_reopt_top_share),
        },
    }
    write_json(manifest_path, manifest)
    return manifest



def run_full_paper_pipeline(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    prepare = prepare_simulation_grid(config, force=force)
    training = train_all_seed_variants(config, force=force)
    evaluation = run_rolling_latency_evaluation(config, force=force)
    payload = {
        'prepare': prepare,
        'training': training,
        'evaluation': evaluation,
    }
    write_json(config.result_dir / 'run_full_paper_pipeline.json', payload)
    return payload
