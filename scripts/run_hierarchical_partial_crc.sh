#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
SEED="${SEED:-151,152,153}"
SEEDS="${SEEDS:-$SEED}"
DATA_DIR="${DATA_DIR:-data/acquire_10k}"
AGGREGATE_PATH="${AGGREGATE_PATH:-${DATA_DIR}/customer_offer_aggregates_10k.csv}"
TRAIN_HISTORY_PATH="${TRAIN_HISTORY_PATH:-${DATA_DIR}/trainHistory_10k.csv.gz}"
OFFERS_PATH="${OFFERS_PATH:-${DATA_DIR}/offers_10k.csv.gz}"
BUDGETS="${BUDGETS:-2760000,5480000,7550000}"
python scripts/prepare_acquire_valued_shoppers.py \
  --aggregate-path "$AGGREGATE_PATH" \
  --train-history-path "$TRAIN_HISTORY_PATH" \
  --offers-path "$OFFERS_PATH" \
  --project-root "$ROOT_DIR" \
  --artifacts-dir artifacts \
  --seeds "$SEEDS" \
  --household-limit 10000
python main.py \
  --mode run-hierarchical \
  --project-root "$ROOT_DIR" \
  --seeds "$SEEDS" \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 1,3,7 \
  --budgets "$BUDGETS" \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300 \
  --partial-reopt-score-delta 0.10 \
  --partial-reopt-high-risk-threshold 0.80 \
  --partial-reopt-top-share 0.15 \
  --alpha-grid 0.05,0.10,0.20 \
  --conformal-min-cal-size 200 \
  --hierarchical-max-call-ratio 0.15 \
  --force
