#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-.}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_acquire_10k_seed151}"
SEED="${SEED:-151}"
DATA_DIR="${DATA_DIR:-data/acquire_10k}"
AGGREGATE_PATH="${AGGREGATE_PATH:-${DATA_DIR}/customer_offer_aggregates_10k.csv}"
TRAIN_HISTORY_PATH="${TRAIN_HISTORY_PATH:-${DATA_DIR}/trainHistory_10k.csv.gz}"
OFFERS_PATH="${OFFERS_PATH:-${DATA_DIR}/offers_10k.csv.gz}"

# 캡디논문10의 외부검증 선택 압박(약 1.6% / 3.2% / 4.4%)을 1만명 표본에 맞춰 유지.
# 목표 선택 고객수는 대략 160 / 320 / 440명이며,
# sample_manifest + budget_manifest 기준 권장 예산은 아래와 같다.
BUDGETS="${BUDGETS:-2760000,5480000,7550000}"

python scripts/prepare_acquire_valued_shoppers.py \
  --aggregate-path "$AGGREGATE_PATH" \
  --train-history-path "$TRAIN_HISTORY_PATH" \
  --offers-path "$OFFERS_PATH" \
  --project-root "$PROJECT_ROOT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --seeds "$SEED" \
  --household-limit 10000

python main.py \
  --mode run-paper \
  --project-root "$PROJECT_ROOT" \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --seeds "$SEED" \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets "$BUDGETS" \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300
