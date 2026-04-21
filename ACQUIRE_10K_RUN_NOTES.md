# Acquire Valued Shoppers 10k / 1-seed setup

이 디렉토리는 기존 dunnhumby 실험 베드 구조를 유지한 채, Kaggle `Acquire Valued Shoppers Challenge` 데이터를 10,000명 표본으로 맞춘 실행본입니다.

## 포함된 데이터
- `data/acquire_10k/customer_offer_aggregates_10k.csv`
- `data/acquire_10k/trainHistory_10k.csv.gz`
- `data/acquire_10k/offers_10k.csv.gz`
- `data/acquire_10k/sample_manifest.json`
- `data/acquire_10k/budget_manifest.json`

## 표본 추출
- 총 160,057명 전체 train 고객에서 10,000명을 고정 추출
- 난수 seed: `20260422`
- 층화 기준: `offervalue × repeater_flag`

## 실행 seed
- 단일 seed: `151`
- 목적: 외부 보강 실험을 빠르게 돌리기 위한 1-seed 실행
- 주의: 최종 robustness 주장에는 3-seed가 더 안전하지만, 외부 보강 확인용으로는 1-seed도 사용 가능

## 저/중/고 예산 기준
캡디논문10 외부검증의 선택 압박을 그대로 유지하도록 고객 비율 기준을 맞췄습니다.

- low: 약 1.6% 선택 압박 → 목표 선택 고객수 약 160명
- mid: 약 3.2% 선택 압박 → 목표 선택 고객수 약 320명
- high: 약 4.4% 선택 압박 → 목표 선택 고객수 약 440명

권장 예산:
- low: `2,760,000`
- mid: `5,480,000`
- high: `7,550,000`

예산은 `budget_manifest.json`의 proxy ranking 기준 상위 고객들의 예상 coupon cost 합으로 정했습니다.

## 실행
```bash
bash scripts/run_acquire_10k_one_seed.sh
```

또는 수동 실행:
```bash
python scripts/prepare_acquire_valued_shoppers.py \
  --aggregate-path data/acquire_10k/customer_offer_aggregates_10k.csv \
  --train-history-path data/acquire_10k/trainHistory_10k.csv.gz \
  --offers-path data/acquire_10k/offers_10k.csv.gz \
  --project-root . \
  --artifacts-dir artifacts_acquire_10k_seed151 \
  --seeds 151 \
  --household-limit 10000

python main.py \
  --mode run-paper \
  --project-root . \
  --artifacts-dir artifacts_acquire_10k_seed151 \
  --seeds 151 \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets 2760000,5480000,7550000 \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300
```
