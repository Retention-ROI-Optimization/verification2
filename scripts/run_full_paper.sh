#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python main.py \
  --mode run-paper \
  --project-root "$ROOT_DIR" \
  --seeds 41,42,43\
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --bootstrap-iterations 1000
