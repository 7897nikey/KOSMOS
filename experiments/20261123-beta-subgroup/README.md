# 20261123-beta-subgroup (보정 β의 서브그룹 단위)

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-23 |
| 상태·결론 | `meta.json` |
| 데이터 | `research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz`(K-EXAONE 2.5·DeepSeek Ultrafast·Haiku 5.5 추론 끔, k=100), `raw/v2/deepseek_eco_p1.jsonl`·`deepseek_eco_low_p1.jsonl`(DeepSeek Eco 추론 끔·low, 표본 0~49). 모두 원문 문체, 페르소나 없음·한국 성인·성별×연령 10셀. 해시는 `out/run.json` |

## 질문 (돌리기 전에 쓴다)

`20261119-beta-scope`는 β를 묶는 단위를 페르소나 없음·한국 성인 두 단계 안에서만 비교했다. 성별×연령 10셀(이하 10셀)까지 넣으면 β를 어느 단위로 정해야 하는가. 셀마다 따로 고르면 셀 정답(사람 표본 39~281명)의 잡음에 맞춰질 수 있고, 너무 넓게 묶으면 설정 사이 차이를 놓친다.

## 가설과 판정 기준 (돌리기 전에 쓴다)

방향 예측은 두지 않는다. test 49문항은 쓰지 않는다.

**보정 식**: p ∝ (개수+1)^β. 격자는 0.05~1.5, 0.005 간격이다. β는 그룹의 calibration 148문항 목적함수가 최소인 값이다.

**목적함수**: 그룹 안 단계(페르소나 없음, 한국 성인, 10셀)를 같은 가중치로 평균한다. 10셀 안은 `cells.csv`의 `pop_share`로 가중한다.

**설정 5개**: K-EXAONE 2.5(k=100), DeepSeek Ultrafast 추론 끔(k=100), Haiku 5.5 추론 끔(k=100), DeepSeek Eco 추론 끔(k=50), DeepSeek Eco 추론 low(k=50).

**β를 묶는 단위**

| 이름 | 그룹 | β 개수 |
|---|---|---|
| 셀마다 | 설정 × 페르소나(셀 하나하나) | 60 |
| 단계별 | 설정 × 단계(10셀이 β 하나 공유) | 15 |
| 페르소나 유무 | 설정 × {페르소나 없음 / 한국 성인+10셀} | 10 |
| 설정별 | 설정 | 5 |
| 전체 하나 | 모두 | 1 |

**판정 기준** (beta-scope와 같다)

| 기준 | 정의 | 통과 조건 |
|---|---|---|
| 손실 | validation 49문항 TVD − 셀마다 β의 validation TVD. (설정, 단계) 15칸 평균이고 10셀은 pop_share 가중. 문항끼리 짝지은 부트스트랩 2,000회 | 평균 ≤ 0.005 |
| 결론 불변 | 효과 19개의 validation TVD 차이에서 부호나 유의성(95% 구간이 0을 포함하는지)이 셀마다 β와 달라지는 개수 | ≤ 3개 |
| 옮겨 쓰기 | (1) 한국 성인 β를 10셀에 그대로 쓸 때, (2) 다른 설정들의 같은 단계에서 고른 β를 쓸 때의 손실 | 평균 ≤ 0.005 |

**효과 19개** (validation, 낮을수록 사람에 가까움)
- 페르소나: 페르소나 없음 → 한국 성인 (설정별 5개)
- 셀 페르소나: 한국 성인 분포 → 10셀 분포. 둘 다 셀 정답과 비교하고 pop_share로 가중한다 (설정별 5개).
- 추론: DeepSeek Eco 끔 → low (페르소나 없음, 한국 성인, 10셀; 3개)
- 모델: Haiku − K-EXAONE, DeepSeek Ultrafast − K-EXAONE, Haiku − DeepSeek Ultrafast (한국 성인, 10셀; 6개)

통과하는 단위 중 가장 넓은 것을 **β 규칙**으로 정한다. 이후 실험(추론 설명, 인칭·화계 확정, KGSS 회차)은 이 규칙을 쓴다.

**보조(판정에 안 씀)**
- 5겹 교차검증: `items_246.csv`의 `group_cv_fold`를 쓰고, test를 뺀 197문항에서 한 겹을 빼고 β를 골라 그 겹에서 잰다. 단위별 손실을 함께 보고한다.
- 사람 정답 잡음 기준선: 셀 정답을 n_eff명으로 다시 뽑았을 때 원래 정답과의 TVD 평균. 셀마다 β가 이 잡음보다 작은 차이를 쫓는지 본다.

## 방법

`run.py` 하나다. 개수 불러오기는 `20261118-reasoning-effect/run.py`의 `load_counts`, 보정·TVD는 `20261119-register-effect/run.py`의 `prob`·`tvd`, 단위 비교 틀은 `20261119-beta-scope/run.py`를 복사했다(출처 주석 있음).

출력:
- `out/betas.csv`
- `out/loss.csv`, `out/loss_summary.csv`
- `out/effects.csv`, `out/effects_changed.csv`
- `out/transfer_summary.csv`
- `out/cv_summary.csv`
- `out/noise.csv`
- `out/run.log`, `out/run.json`

## 결과 (돌린 뒤 쓴다)

## 결론

## 다음
