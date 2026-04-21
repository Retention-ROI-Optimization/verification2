from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external_datasets.acquire_valued_shoppers import ImportConfig, import_acquire_valued_shoppers
from src.paper_latency.config import parse_int_list


DEFAULT_SEEDS: tuple[int, ...] = (151, 152, 153)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert Kaggle Acquire Valued Shoppers files into this experiment bed\'s raw_grid schema.')
    parser.add_argument('--aggregate-path', required=True, help='Path to customer_offer_aggregates.csv')
    parser.add_argument('--train-history-path', required=True, help='Path to trainHistory.csv.gz')
    parser.add_argument('--offers-path', required=True, help='Path to offers.csv.gz')
    parser.add_argument('--project-root', default='.', help='Experiment project root')
    parser.add_argument('--artifacts-dir', default='artifacts', help='Artifact root directory relative to project root')
    parser.add_argument('--seeds', default=','.join(str(x) for x in DEFAULT_SEEDS), help='Seed directories to populate, e.g. 151,152,153')
    parser.add_argument('--household-limit', type=int, default=None, help='Optional cap for a tractable subset, e.g. 30000')
    parser.add_argument('--snapshot-frequency-days', type=int, default=7)
    parser.add_argument('--dormant-inactivity-days', type=int, default=14)
    parser.add_argument('--churn-inactivity-days', type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ImportConfig(
        project_root=Path(args.project_root).resolve(),
        aggregate_path=Path(args.aggregate_path).resolve(),
        train_history_path=Path(args.train_history_path).resolve(),
        offers_path=Path(args.offers_path).resolve(),
        seeds=parse_int_list(args.seeds, DEFAULT_SEEDS),
        household_limit=args.household_limit,
        snapshot_frequency_days=int(args.snapshot_frequency_days),
        dormant_inactivity_days=int(args.dormant_inactivity_days),
        churn_inactivity_days=int(args.churn_inactivity_days),
        artifacts_dir=str(args.artifacts_dir),
    )
    manifest = import_acquire_valued_shoppers(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
