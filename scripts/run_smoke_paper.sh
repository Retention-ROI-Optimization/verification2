#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python main.py \
  --mode run-paper \
  --project-root "$ROOT_DIR" \
  --seeds 41 \
  --scenario-families complaint-heavy,promotion-heavy \
  --latencies 0,1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --decision-week-limit 2 \
  --bootstrap-iterations 100 \
  --training-landmarks 4
