# 20261115-baseline-calibration

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-15 |
| 상태·결론 | `meta.json` |
| 데이터 | `research/design_v2_20261112/dataset/kosmos_long.csv.gz` (해시는 `out/run.json`) |

## 질문 (돌리기 전에 쓴다)

지금 모인 데이터(EXAONE k≈72, DeepSeek·Haiku k=100)에서 설계 v2 보정(라플라스 α=1, p ∝ (c+1)^β)으로 고른 β*와 validation TVD가 얼마인가. 이후 실험이 비교할 기준선이다.

## 가설과 판정 기준 (돌리기 전에 쓴다)

- 가설: 없음(기준선 측정). 대신 재현 점검을 한다 — DeepSeek·Haiku β*가 `analysis/out_calib/frozen_beta.json`(laplace)과 같아야 한다.
- 쓰는 문항 분할: calibration 148(β 고르기) / validation 49(확인). test는 쓰지 않는다.

## 방법

템플릿 `run.py` 그대로. β 그리드 0.05~1.5(0.005 간격), 모델×단계(c0/L0/L1)마다 calibration 평균 TVD 최소. L1은 10셀을 문항별로 평균한 뒤 문항 평균.

## 결과 (돌린 뒤 쓴다)

β* (`out/run.json`): DeepSeek 0.315 / 0.315 / 0.31, Haiku 0.24 / 0.245 / 0.25 (c0 / L0 / L1) — `frozen_beta.json`과 6개 모두 같다. EXAONE 0.335 / 0.38 / 0.385.

validation TVD, β* (`out/validation_tvd_beta_star.csv`)

| 단계 | EXAONE | DeepSeek | Haiku | 균등 |
|---|---:|---:|---:|---:|
| c0 | 0.298 | 0.282 | 0.300 | 0.335 |
| L0 | 0.243 | 0.261 | 0.293 | 0.335 |
| L1 | 0.248 | 0.285 | 0.285 | 0.353 |

β=1(평활만, `out/validation_tvd_beta1.csv`)이면 세 모델 모두 균등보다 나쁘다(L1 EXAONE 0.369, DeepSeek 0.443, Haiku 0.486).

주의: 표본 수가 다르다(EXAONE 69~73, 나머지 100). 표본 수를 맞춘 비교는 `analysis/out_exaone_interim/REPORT.md`가 기준이다(그쪽 DeepSeek L0는 0.270).

## 결론

보정 프레임이 CPU용 DataFrame에서 그대로 재현된다. 보정 후 L0·L1에서 EXAONE이 가장 낮다.

## 다음

- EXAONE k=100이 모이면 `build_dataset.py`를 다시 돌리고 이 실험을 다시 실행한다.
