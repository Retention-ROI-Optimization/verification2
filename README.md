# Retention Latency Paper Experiment Bed

이 프로젝트는 **이탈 점수 freshness / latency가 예산 제약형 리텐션 정책의 가치와 안정성에 어떤 영향을 주는지** 반복적으로 평가하기 위한 실험 베드입니다.  
특히 논문 본문에서 다루는 두 축을 재현할 수 있도록 구성되어 있습니다.

1. **메인 실험 (`run-paper`)**  
   stale 점수, stronger-but-stale, weaker-but-fresh, full-refresh를 같은 의사결정 엔진 위에서 비교합니다.
2. **세타 민감도 실험 (`run-theta-sensitivity`)**  
   부분 재최적화(partial re-optimization)에서 \\(\theta\\)를 바꿔도 핵심 결론이 유지되는지 점검합니다.

기존 연구용 README는 `README_FOR_RESEARCH.md`로 옮겨 두었습니다. 이 파일은 **처음 실행하는 사람도 바로 따라갈 수 있도록** 메인 실험과 세타 테스트를 중심으로 다시 정리한 안내서입니다.

---

## 1. 빠른 시작

### 1-1. 환경 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1-2. 가장 먼저 smoke test

```bash
bash scripts/run_smoke_paper.sh
```

이 단계는 전체 재현 전에 경로, 의존성, 결과 파일 생성 여부를 빠르게 점검하기 위한 최소 실행입니다.

### 1-3. 논문용 메인 실험

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

### 1-4. 세타 민감도 실험

```bash
python main.py \
  --mode run-theta-sensitivity \
  --project-root . \
  --seeds 41,42,43 \
  --scenario-families complaint-heavy,promotion-heavy,dormancy-heavy,seasonal-shift \
  --latencies 1,3,7 \
  --budgets 2640000,7250000,11530000 \
  --burn-in-weeks 12 \
  --training-landmarks 12 \
  --decision-week-limit 16 \
  --bootstrap-iterations 300 \
  --theta-grid 0.05,0.10,0.15 \
  --partial-reopt-high-risk-threshold 0.80 \
  --partial-reopt-top-share 0.15
```

같은 명령은 각각 `scripts/run_full_paper.sh`, `scripts/run_theta_sensitivity.sh`에도 들어 있습니다.

---

## 2. 프로젝트 구조

```text
Experiment/
├── main.py
├── README.md
├── README_FOR_RESEARCH.md
├── requirements.txt
├── scripts/
│   ├── run_smoke_paper.sh
│   ├── run_full_paper.sh
│   └── run_theta_sensitivity.sh
├── results/
│   ├── README.md
│   ├── 01_summary_clean.csv
│   ├── 02_by_scenario_policy.csv
│   ├── 03_policy_overall.csv
│   ├── 04_latency_effect.csv
│   ├── 05_budget_effect.csv
│   ├── 06_block_stats.csv
│   └── 07_policy_vs_full_refresh.csv
└── src/
    ├── simulator/
    ├── features/
    ├── optimization/
    └── paper_latency/
```

실행 중 생성되는 핵심 산출물은 아래 경로에 쌓입니다.

```text
artifacts/
├── raw_grid/
├── feature_cache/
├── models/
└── results/
    ├── training/
    └── paper_latency/
        ├── block_level_metrics.csv
        ├── summary_metrics.csv
        └── theta_sensitivity/
            └── 0p05__0p1__0p15/
                ├── theta_block_level_metrics.csv
                ├── theta_summary_by_latency.csv
                ├── theta_summary_overall.csv
                └── manifest.json
```

---

## 3. 메인 실험이 무엇을 검증하는가

메인 실험의 질문은 단순히 "stale 점수가 평균 정책 가치(policy value)를 얼마나 떨어뜨리는가"가 아닙니다.  
이 실험은 더 나아가, **점수 freshness 저하가 실제 타깃 고객 집합과 개입 시점을 얼마나 흔드는가**를 함께 봅니다.

즉, 본 실험은 다음 두 층위를 동시에 측정합니다.

1. **가치 층위**  
   stale 점수를 쓴 정책이 fresh 기준 정책에 비해 얼마나 손해를 보는가
2. **안정성 층위**  
   stale 점수 때문에 타깃 고객이 얼마나 바뀌고, 개입 타이밍이 얼마나 어긋나는가

이를 위해 동일한 예산 최적화 엔진 위에서 아래 정책들을 비교합니다.

- `base-stale`: 기본 모델의 stale 점수 사용
- `full-refresh`: 같은 기본 모델이지만 decision 시점의 fresh 점수 사용
- `stronger-but-stale`: 더 강한 모델이지만 stale 점수 사용
- `weaker-but-fresh`: 더 약한 모델이지만 fresh 점수 사용

이 비교를 통해 다음 논점을 동시에 확인할 수 있습니다.

- 점수 지연이 생기면 평균 가치가 조금만 변해도 타깃 정합성은 크게 무너질 수 있는가
- 모델 강도 향상이 freshness 손실을 상쇄할 수 있는가
- stale 정책의 손실이 단순 value loss가 아니라 action queue 교란으로도 나타나는가

---

## 4. 메인 실험 설정

### 4-1. 비교 축

- **Seed**: `41, 42, 43`
- **Scenario family**
  - `complaint-heavy`
  - `promotion-heavy`
  - `dormancy-heavy`
  - `seasonal-shift`
- **Latency**: `0, 1, 3, 7일`
- **Budget**: `2640000, 7250000, 11530000`
- **Burn-in**: `12주`
- **Decision weeks**: 이후 `16개 주`
- **Bootstrap iterations**: `300`

즉, 논문용 compact run은 **3 seeds × 4 scenario families × 16 decision weeks × 3 budgets × 여러 정책 비교**를 반복하는 구조입니다.

### 4-2. 왜 이 구성을 쓰는가

- **여러 seed**: 시뮬레이터 난수 변동에 덜 민감한 평균 패턴 확보
- **여러 scenario family**: 특정 고객 행동 패턴 하나에만 맞는 결론을 피하기 위함
- **여러 latency**: freshness 손실의 크기를 단계별로 확인
- **여러 budget**: 예산 제약이 강할수록 freshness 비용이 어떻게 달라지는지 확인
- **rolling decision weeks**: 단일 시점이 아니라 연속 의사결정 구간에서 결과를 측정

---

## 5. 메인 실험 실행 순서

메인 파이프라인은 내부적으로 아래 순서를 따릅니다.

### 단계 1. 시뮬레이션 그리드 준비

`prepare_simulation_grid()`가 seed별 데이터셋을 만듭니다.

생성 또는 재사용되는 원시 데이터 예시는 다음과 같습니다.

- `customers.csv`
- `events.csv`
- `orders.csv`
- `state_snapshots.csv`
- `campaign_exposures.csv`
- `treatment_assignments.csv`
- `customer_summary.csv`
- `cohort_retention.csv`

### 단계 2. 모델 variant 학습

`train_all_seed_variants()`가 seed별로 세 모델을 학습합니다.

- `base`
- `stronger`
- `weaker`

학습 산출물은 `artifacts/models/seed_xx/` 아래에 저장됩니다.

### 단계 3. rolling latency evaluation

각 decision date마다 fresh 스냅샷과 stale 스냅샷을 만들고, 동일한 엔진으로 정책 선택을 수행한 뒤 fresh 기준 정책과 비교합니다.

즉, 메인 실험의 핵심은 아래 비교입니다.

- **fresh 기준 정책**: decision 시점의 최신 점수 사용
- **candidate 정책**: stale 점수 혹은 stronger/weaker 점수 사용
- **비교 결과**: 가치 차이와 타깃 정합성 차이를 함께 기록

---

## 6. 메인 실험 결과 파일

### 6-1. 원본 결과

#### `artifacts/results/paper_latency/block_level_metrics.csv`
가장 세밀한 결과 파일입니다.  
한 행이 하나의 **seed × scenario_family × decision_date × budget × policy_kind × latency** 비교 블록입니다.

대표 컬럼:

- `seed`
- `scenario_family`
- `decision_date`
- `budget`
- `policy_kind`
- `latency_days`
- `policy_value`
- `stale_regret`
- `relative_loss`
- `target_overlap`
- `missed_at_risk`
- `window_miss_rate`
- `partial_reopt_optimization_call_ratio`
- `partial_reopt_regret_recovery_ratio`
- `partial_reopt_full_refresh_value_ratio`

#### `artifacts/results/paper_latency/summary_metrics.csv`
위 block-level 결과를 그룹별로 묶어 평균과 95% bootstrap CI를 계산한 파일입니다.

그룹 축:

- `scenario_family`
- `budget`
- `policy_kind`
- `latency_days`

### 6-2. 사람이 읽기 쉽게 정리한 결과

`results/README.md`와 아래 CSV들은 원본 결과를 요약해 둔 보조 자료입니다.

- `01_summary_clean.csv`
- `02_by_scenario_policy.csv`
- `03_policy_overall.csv`
- `04_latency_effect.csv`
- `05_budget_effect.csv`
- `06_block_stats.csv`
- `07_policy_vs_full_refresh.csv`

논문 본문용 표를 만들 때는 보통 이 폴더를 먼저 확인한 뒤, 필요하면 `artifacts/results/paper_latency/`의 원본으로 내려가면 됩니다.

---

## 7. 메인 실험 지표 해석

### `policy_value`
해당 정책이 의사결정 엔진 아래에서 얻은 정책 가치입니다.  
가장 직관적인 성능 지표이지만, 이것만 보면 freshness 문제를 과소평가할 수 있습니다.

### `stale_regret`
fresh 기준 정책 대비 가치 차이를 나타냅니다.  
일부 조건에서 음수가 나올 수 있는데, 이는 stale 정책이 정말 더 우수하다는 뜻이 아니라 **stale 정보로 평가가 낙관적으로 보일 가능성**도 함께 시사합니다.

### `relative_loss`
가치 차이를 상대 비율로 본 값입니다.  
예산 크기나 정책 간 손실 규모를 비교할 때 편합니다.

### `target_overlap`
fresh 기준 정책이 선택한 고객 집합과 candidate 정책이 선택한 고객 집합이 얼마나 겹치는지를 봅니다.  
평균 가치가 비슷해도 overlap이 낮으면 실제 운영에서 다른 고객에게 쿠폰을 보내고 있다는 뜻입니다.

### `missed_at_risk`
fresh 기준에서는 타깃이었어야 할 고위험 고객을 얼마나 놓쳤는지 나타냅니다.  
정책 가치만으로는 잘 드러나지 않는 **누락 비용**을 보여줍니다.

### `window_miss_rate`
개입 시점을 놓친 비율입니다.  
장기 지연의 비용은 종종 이 지표에서 더 분명하게 드러납니다.

---

## 8. 세타 민감도 실험이 무엇을 검증하는가

세타 테스트는 부분 재최적화(partial re-optimization)의 선택 규칙이 **특정 \\(\theta\\) 값 하나에만 과도하게 의존하는지**를 확인하기 위한 실험입니다.

부분 재최적화의 기본 아이디어는 다음과 같습니다.

- 시점 \\(t\\)의 점수 \\(s_t\\)와 과거 점수 \\(s_{t-L}\\) 사이의 절대 변화량을 계산한다.
- 그 변화량이 임계값 \\(\theta\\)를 넘는 고객만 다시 최적화한다.
- 나머지 고객은 stale 정책 결과를 그대로 유지한다.

이때 본 프로젝트의 관심은 **"최적 \\(\theta\\)를 찾는 것" 자체가 아닙니다.**  
더 중요한 질문은 다음입니다.

> \\(\theta\\)를 하나의 운영 임계값으로 고정해도, 재최적화 호출 비율을 크게 줄이면서 full-refresh에 가까운 정책 가치를 유지할 수 있는가?

그래서 세타 테스트는 **최적 파라미터 탐색 실험**이라기보다, **운영형 선택 규칙의 실용 가능성 검증 실험**에 가깝습니다.

---

## 9. 세타 민감도 실험 설정

### 9-1. 고정되는 값

스크립트 기준으로 아래 값은 고정됩니다.

- `--partial-reopt-high-risk-threshold 0.80`
- `--partial-reopt-top-share 0.15`

즉, high-risk 보완 규칙과 top-share 조건은 유지한 채, **score delta threshold 역할을 하는 \\(\theta\\)만 바꿔 봅니다.**

### 9-2. 변화시키는 값

- `--theta-grid 0.05,0.10,0.15`

코드상 이 값은 `partial_reopt_score_delta`에 대응합니다.  
즉, `run-theta-sensitivity`는 사실상 **score delta threshold sweep**입니다.

### 9-3. 왜 latency는 1, 3, 7만 보는가

- `L=0`은 fresh 기준선이므로 부분 재최적화의 필요성이 거의 없습니다.
- 따라서 세타 민감도에서는 **실제로 stale 문제가 생기는 구간**인 `1, 3, 7일`에 집중합니다.

---

## 10. 세타 실험 결과 파일

세타 실험 결과는 아래 경로에 저장됩니다.

```text
artifacts/results/paper_latency/theta_sensitivity/0p05__0p1__0p15/
```

### `theta_block_level_metrics.csv`
가장 세밀한 원본입니다.  
한 행이 하나의 **seed × scenario_family × decision_date × budget × latency × theta** 조합입니다.

대표 컬럼:

- `theta`
- `latency_days`
- `base_policy_value`
- `base_stale_regret`
- `base_relative_loss`
- `base_target_overlap`
- `base_missed_at_risk`
- `base_window_miss_rate`
- `partial_reopt_policy_value`
- `partial_reopt_stale_regret`
- `partial_reopt_regret_recovery_ratio`
- `partial_reopt_full_refresh_value_ratio`
- `partial_reopt_optimization_call_ratio`

### `theta_summary_by_latency.csv`
\(\theta\)와 latency별 평균 및 95% bootstrap CI를 제공합니다.  
즉, **"같은 지연일에서 \\(\theta\\)를 바꾸면 call ratio와 value ratio가 어떻게 변하는가"** 를 가장 쉽게 볼 수 있는 파일입니다.

### `theta_summary_overall.csv`
latency를 통합한 전체 평균 및 95% bootstrap CI입니다.  
논문 본문에서 **"핵심 결론이 \\(\theta\\) 변화에도 유지되는가"** 를 간단히 보여줄 때 가장 쓰기 좋습니다.

### `manifest.json`
실험에 사용된 설정과 출력 파일 경로를 저장합니다.

---

## 11. 세타 실험에서 특히 볼 지표

### `partial_reopt_optimization_call_ratio`
전체 고객 중 실제로 재최적화를 다시 수행한 비율입니다.  
값이 낮을수록 계산 비용을 덜 쓴 것입니다.

### `partial_reopt_full_refresh_value_ratio`
부분 재최적화 결과의 정책 가치가 full-refresh 가치의 몇 % 수준인지 나타냅니다.  
1에 가까울수록, 전체를 다시 최적화하지 않고도 거의 같은 성능을 유지한 것입니다.

### `partial_reopt_regret_recovery_ratio`
stale 정책이 갖고 있던 regret을 부분 재최적화가 얼마나 회복했는지를 보여줍니다.  
운영적으로는 **적은 호출로 얼마만큼 손실을 되찾는가**를 보는 지표입니다.

---

## 12. 권장 해석 순서

### 메인 실험 해석 순서

1. `03_policy_overall.csv`로 전체 평균 흐름 확인
2. `04_latency_effect.csv`로 latency가 커질수록 어떤 지표가 먼저 흔들리는지 확인
3. `05_budget_effect.csv`로 예산에 따라 손실 구조가 달라지는지 확인
4. 필요하면 `artifacts/results/paper_latency/summary_metrics.csv`로 95% CI 확인
5. 더 세부 조합이 필요하면 `block_level_metrics.csv`로 내려가기

### 세타 실험 해석 순서

1. `theta_summary_overall.csv`로 \(\theta\)별 전체 평균 비교
2. `theta_summary_by_latency.csv`로 지연일별 민감도 확인
3. `theta_block_level_metrics.csv`에서 특정 scenario 또는 budget 조합 재검토

---

## 13. 재실행 시 유용한 정리 규칙

### 결과만 다시 만들고 싶을 때

```bash
rm -rf artifacts/results/paper_latency
mkdir -p artifacts/results/paper_latency
```

### feature cache까지 비우고 싶을 때

```bash
rm -rf artifacts/feature_cache
mkdir -p artifacts/feature_cache
```

### 이미 생성된 seed별 raw data / model을 재사용하고 싶다면

`force` 옵션 없이 다시 실행하면 기존 산출물을 우선 재사용합니다.

---

## 14. 파일 역할 요약

- `README.md`: 현재 프로젝트 사용자를 위한 메인 안내서
- `README_FOR_RESEARCH.md`: 기존 연구용 README 보존본
- `results/README.md`: 요약 CSV 해설서
- `scripts/run_full_paper.sh`: 메인 논문 실험 실행 스크립트
- `scripts/run_theta_sensitivity.sh`: 세타 민감도 실행 스크립트



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
