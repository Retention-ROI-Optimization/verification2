#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_acquire_vs}"
SEEDS="${SEEDS:-151,152,153}"
HOUSEHOLD_LIMIT="${HOUSEHOLD_LIMIT:-30000}"
AGGREGATE_PATH="${AGGREGATE_PATH:-/absolute/path/to/customer_offer_aggregates.csv}"
TRAIN_HISTORY_PATH="${TRAIN_HISTORY_PATH:-/absolute/path/to/trainHistory.csv.gz}"
OFFERS_PATH="${OFFERS_PATH:-/absolute/path/to/offers.csv.gz}"

# Full uploaded file 기준 저/중/고 예산(5% / 15% / 30% 타깃 share)
# household_limit를 쓰면 아래 예산도 동일 비율로 축소해 사용하는 것이 좋습니다.
BUDGETS="${BUDGETS:-103837500,311120000,622660000}"

python scripts/prepare_acquire_valued_shoppers.py \
  --aggregate-path "$AGGREGATE_PATH" \
  --train-history-path "$TRAIN_HISTORY_PATH" \
  --offers-path "$OFFERS_PATH" \
  --project-root "$PROJECT_ROOT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --seeds "$SEEDS" \
  --household-limit "$HOUSEHOLD_LIMIT"

python main.py \
  --mode run-paper \
  --project-root "$PROJECT_ROOT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --seeds "$SEEDS" \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets "$BUDGETS" \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300
