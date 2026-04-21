from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external_datasets.dunnhumby_complete_journey import ImportConfig, import_complete_journey
from src.paper_latency.config import DEFAULT_SEEDS, parse_int_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert dunnhumby The Complete Journey into this experiment bed\'s raw_grid schema.')
    parser.add_argument('--zip-path', required=True, help='Path to dunnhumby_The-Complete-Journey.zip or extracted folder')
    parser.add_argument('--project-root', default='.', help='Experiment project root')
    parser.add_argument('--seeds', default=','.join(str(x) for x in DEFAULT_SEEDS[:3]), help='Seed directories to populate, e.g. 41,42,43')
    parser.add_argument('--household-limit', type=int, default=None, help='Optional cap for debugging smaller imports')
    parser.add_argument('--snapshot-frequency-days', type=int, default=7)
    parser.add_argument('--dormant-inactivity-days', type=int, default=14)
    parser.add_argument('--churn-inactivity-days', type=int, default=30)
    parser.add_argument('--start-date', default='2023-01-01', help='Anchor calendar date used for DAY=1')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ImportConfig(
        project_root=Path(args.project_root).resolve(),
        source_path=Path(args.zip_path).resolve(),
        seeds=parse_int_list(args.seeds, DEFAULT_SEEDS[:3]),
        household_limit=args.household_limit,
        snapshot_frequency_days=int(args.snapshot_frequency_days),
        dormant_inactivity_days=int(args.dormant_inactivity_days),
        churn_inactivity_days=int(args.churn_inactivity_days),
        start_date=str(args.start_date),
    )
    manifest = import_complete_journey(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
