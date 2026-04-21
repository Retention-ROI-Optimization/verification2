# Retention Latency Paper Experiment Bed

이 디렉토리는 **이탈 위험 점수 freshness / latency가 예산 제약형 리텐션 정책에 어떤 영향을 주는지** 반복 평가하기 위한 실험 전용 프로젝트입니다.

기본적으로 아래 3단계 실행 경로를 권장합니다.

1. **Smoke test**: 1 seed × 2 scenario families × 2 decision weeks
2. **Validated mid-run**: 2 seeds × 4 scenario families × 5 decision weeks
3. **Compact paper run**: 3 seeds × 4 scenario families × 16 decision weeks

`run_full_paper.sh`가 구현상 가능하긴 하지만 시간이 매우 오래 걸릴 수 있으므로 **논문 본문용 결과는 compact paper run을 기본 권장값**으로 둡니다.

---

## 1. 프로젝트 구조

```text
Experiment/
├── main.py
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_smoke_paper.sh
│   └── run_full_paper.sh
└── src/
    ├── simulator/
    ├── features/
    ├── optimization/
    └── paper_latency/
```

주요 산출물은 아래 경로에 생성됩니다.

```text
artifacts/
├── raw_grid/
├── feature_cache/
├── models/
└── results/
    ├── training/
    └── paper_latency/
        ├── block_level_metrics.csv
        └── summary_metrics.csv
```

---

## 2. 이번 실험에서 실제로 사용한 비교 축

- **Scenario family 4개**
  - `complaint-heavy`
  - `promotion-heavy`
  - `dormancy-heavy`
  - `seasonal-shift`
- **Latency 4개**: `0, 1, 3, 7일`
- **Budget 3개**: `2640000, 7250000, 11530000`
- **Policy comparison**
  - `base-stale`
  - `stronger-but-stale`
  - `weaker-but-fresh`
  - `full-refresh`
- **핵심 지표**
  - `policy_value`
  - `stale_regret`
  - `relative_loss`
  - `target_overlap`
  - `missed_at_risk`
  - `window_miss_rate`
- **보조 지표**
  - `partial_reopt_regret_recovery_ratio`
  - `partial_reopt_full_refresh_value_ratio`
  - `partial_reopt_optimization_call_ratio`

---


## 3. 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. 실행 권장 순서

### 4-1. Smoke test

가장 먼저 구조 점검용으로 아래를 실행합니다.

```bash
bash scripts/run_smoke_paper.sh
```

직접 실행 명령은 아래와 같습니다.

```bash
python main.py \
  --mode run-paper \
  --project-root . \
  --seeds 41 \
  --scenario-families complaint-heavy,promotion-heavy \
  --latencies 0,1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --decision-week-limit 2 \
  --bootstrap-iterations 100 \
  --training-landmarks 4
```

### 4-2. Validated mid-run

이 설정은 실제로 통과시킨 **중간 검증 규모**입니다.

```bash
python main.py \
  --mode run-paper \
  --project-root . \
  --seeds 41,42 \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --burn-in-weeks 12 \
  --training-landmarks 4 \
  --decision-week-limit 5 \
  --bootstrap-iterations 200
```

이 실행이 끝나면 `summary_metrics.csv`의 `block_count`가 보통 **10**(= 2 seeds × 5 weeks) 수준으로 나옵니다.

### 4-3. Compact paper run (권장)

논문 본문용으로는 full run 대신 아래 규모를 기본 권장합니다.

```bash
python main.py \
  --mode run-paper \
  --project-root . \
  --seeds 41,42,43 \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 0,1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300
```

이 설정이면 **multiple seeds × multiple decision weeks × four scenario families**를 유지하면서도 full run보다 훨씬 현실적으로 돌릴 수 있습니다.

각 seed 디렉토리 안에는 원본 프로젝트와 같은 형식의 아래 CSV들이 들어갑니다.

- `customers.csv`
- `events.csv`
- `orders.csv`
- `state_snapshots.csv`
- `campaign_exposures.csv`
- `treatment_assignments.csv`
- `customer_summary.csv`
- `cohort_retention.csv`


### 4-4. Full reproduction (선택)

```bash
bash scripts/run_full_paper.sh
```

단, CPU 환경에서는 시간이 오래 걸릴 수 있습니다.

---

## 5. 결과 파일 해석

### block_level_metrics.csv
한 줄이 하나의 비교 블록입니다.

기본 축:
- `seed`
- `scenario_family`
- `decision_date`
- `budget`
- `policy_kind`
- `latency_days`

핵심 해석:
- `latency_days=0`의 `base-stale`와 `full-refresh`는 기준선 역할
- `target_overlap < 1`이면 freshness 지연으로 타깃 구성이 바뀐 것
- `window_miss_rate > 0`이면 개입 타이밍을 놓친 고객이 존재함을 의미
- `stale_regret`와 `relative_loss`는 fresh 대비 value 차이를 기록

### summary_metrics.csv
동일 그룹을 묶어 평균과 95% bootstrap CI를 제공합니다.

주요 체크 포인트:
- `block_count`
- `*_mean`
- `*_ci_low`, `*_ci_high`

---

## 6. 재실행 시 어떤 폴더를 지울까

### 그대로 재사용해도 되는 것
- `artifacts/raw_grid/`
- `artifacts/models/seed_41/`, `seed_42/` 등 이미 정상 완료된 seed
- `artifacts/results/training/seed_41/`, `seed_42/`
- `artifacts/feature_cache/` (코드를 바꾸지 않았다면)

### 새 결과만 다시 만들고 싶을 때
```bash
rm -rf artifacts/results/paper_latency
mkdir -p artifacts/results/paper_latency
```

### feature 정의를 바꿨다면
```bash
rm -rf artifacts/feature_cache
mkdir -p artifacts/feature_cache
```

---





## Added in this patch: matched-cost re-optimization baselines

The rolling latency evaluation now exports additional block-level and summary metrics for three same-call-ratio baselines compared against partial re-optimization:

- `random_refresh_*`: randomly chosen customers are re-scored using the fresh score.
- `top_risk_refresh_*`: the highest-risk customers under the stale score are re-scored.
- `top_value_refresh_*`: customers with the largest stale-policy expected incremental profit are re-scored.

Each baseline uses the **same number of re-optimized customers** as the partial re-optimization rule in the same seed/week/scenario/budget/latency block, so the comparison is on *who to refresh*, not on *how many to refresh*.

Additional columns are also exported for partial re-optimization itself:

- `partial_reopt_target_overlap`, `partial_reopt_missed_at_risk`, `partial_reopt_window_miss_rate`, `partial_reopt_relative_loss`
- `partial_reopt_target_overlap_recovery`, `partial_reopt_missed_at_risk_recovery`, `partial_reopt_window_miss_rate_recovery`

Recovery definitions:

- `TO_recovery = (TO_partial - TO_stale) / (1 - TO_stale)`
- `MaR_recovery = (MaR_stale - MaR_partial) / MaR_stale`
- `WMR_recovery = (WMR_stale - WMR_partial) / WMR_stale`
