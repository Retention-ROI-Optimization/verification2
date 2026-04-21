from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper_latency.config import (
    DEFAULT_BUDGETS,
    DEFAULT_LATENCIES,
    DEFAULT_SCENARIO_FAMILIES,
    DEFAULT_SEEDS,
    ExperimentConfig,
    parse_float_list,
    parse_int_list,
    parse_str_list,
)
from src.paper_latency.evaluation import (
    prepare_simulation_grid,
    run_full_paper_pipeline,
    run_rolling_latency_evaluation,
    run_theta_sensitivity,
    train_all_seed_variants,
)


DEFAULT_THETA_GRID: tuple[float, ...] = (0.05, 0.10, 0.15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Paper experiment bed for churn score freshness latency.')
    parser.add_argument('--project-root', default='.', help='Experiment bed root directory')
    parser.add_argument('--seeds', default=','.join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument('--artifacts-dir', default='artifacts', help='Artifact root directory relative to project-root, used for raw_grid/cache/models/results')
    parser.add_argument('--scenario-families', default=','.join(DEFAULT_SCENARIO_FAMILIES))
    parser.add_argument('--latencies', default=','.join(str(x) for x in DEFAULT_LATENCIES))
    parser.add_argument('--budgets', default=','.join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument('--burn-in-weeks', type=int, default=12)
    parser.add_argument('--training-landmarks', type=int, default=12)
    parser.add_argument('--horizon-days', type=int, default=45)
    parser.add_argument('--decision-week-limit', type=int, default=None)
    parser.add_argument('--bootstrap-iterations', type=int, default=1000)
    parser.add_argument('--partial-reopt-score-delta', type=float, default=0.10)
    parser.add_argument('--partial-reopt-high-risk-threshold', type=float, default=0.80)
    parser.add_argument('--partial-reopt-top-share', type=float, default=0.15)
    parser.add_argument('--theta-grid', default=','.join(f'{x:.2f}' for x in DEFAULT_THETA_GRID), help='Comma-separated theta grid for sensitivity sweep; mapped to partial_reopt_score_delta while other partial reopt knobs stay fixed.')
    parser.add_argument('--stronger-vs-weaker-latency-days', type=int, default=3)
    parser.add_argument('--use-learned-dose-response', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--mode', required=True, choices=['prepare-grid', 'train-variants', 'run-rolling', 'run-paper', 'run-theta-sensitivity'])
    return parser



def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig.from_root(
        args.project_root,
        artifacts_dir=args.artifacts_dir,
        seeds=parse_int_list(args.seeds, DEFAULT_SEEDS),
        scenario_families=parse_str_list(args.scenario_families, DEFAULT_SCENARIO_FAMILIES),
        latencies=parse_int_list(args.latencies, DEFAULT_LATENCIES),
        budgets=parse_int_list(args.budgets, DEFAULT_BUDGETS),
        burn_in_weeks=int(args.burn_in_weeks),
        training_landmarks=int(args.training_landmarks),
        horizon_days=int(args.horizon_days),
        bootstrap_iterations=int(args.bootstrap_iterations),
        decision_week_limit=args.decision_week_limit,
        stronger_vs_weaker_latency_days=int(args.stronger_vs_weaker_latency_days),
        partial_reopt_score_delta=float(args.partial_reopt_score_delta),
        partial_reopt_high_risk_threshold=float(args.partial_reopt_high_risk_threshold),
        partial_reopt_top_share=float(args.partial_reopt_top_share),
        use_learned_dose_response=bool(args.use_learned_dose_response),
    )



def main() -> int:
    args = build_parser().parse_args()
    config = resolve_config(args)

    if args.mode == 'prepare-grid':
        payload = prepare_simulation_grid(config, force=args.force)
    elif args.mode == 'train-variants':
        payload = train_all_seed_variants(config, force=args.force)
    elif args.mode == 'run-rolling':
        payload = run_rolling_latency_evaluation(config, force=args.force)
    elif args.mode == 'run-paper':
        payload = run_full_paper_pipeline(config, force=args.force)
    elif args.mode == 'run-theta-sensitivity':
        payload = run_theta_sensitivity(
            config,
            theta_grid=parse_float_list(args.theta_grid, DEFAULT_THETA_GRID),
            force=args.force,
        )
    else:  # pragma: no cover
        raise SystemExit(f'Unsupported mode: {args.mode}')

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
