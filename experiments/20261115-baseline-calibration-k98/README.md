# 20261115-baseline-calibration-k98

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-15 |
| 상태·결론 | `meta.json` (`status`: 진행/완료/중단, `conclusion`: 한 줄) |
| 데이터 | `research/design_v2_20261112/dataset/kosmos_long.csv.gz` (해시는 `out/run.json`) |

## 질문 (돌리기 전에 쓴다)

EXAONE 표본이 프롬프트당 69~73에서 95~99로 늘었다(정주희·허다교·이지은 완료분 + 태우 진행분). 같은 보정(라플라스 α=1, p ∝ (c+1)^β)으로 고른 β*와 validation TVD 기준선이 `20261115-baseline-calibration`(k≈72) 대비 얼마나 달라지는가.

## 가설과 판정 기준 (돌리기 전에 쓴다)

- 가설: 표본이 늘면 EXAONE의 표집 잡음이 줄어 보정 후 TVD가 같거나 약간 낮아진다. DeepSeek·Haiku는 데이터가 같으므로 β*와 TVD가 이전 실험과 완전히 같아야 한다(재현 점검).
- 지지/기각 기준: DeepSeek·Haiku 값이 이전 `out/`과 한 자리도 다르면 파이프라인 오류로 본다. EXAONE은 이전 대비 변화량을 보고하되, 변화가 0.01 미만이면 '표본 수 영향 작음'으로 쓴다.
- EXAONE k는 프롬프트마다 다르다(m1이 아직 진행 중). 그래서 최종 k=100 기준선이 아니라 중간 기준선이다.
- 쓰는 문항 분할: calibration(β·설정 고르기) / validation(확인). test는 쓰지 않는다.

## 방법

템플릿 `run.py` 그대로. 바꾼 것은 데이터뿐이다(`kosmos_build.json`: EXAONE 소스 4개, k 95~99).

## 결과 (돌린 뒤 쓴다)

β* (`out/run.json`, c0 / L0 / L1): EXAONE 0.32 / 0.355 / 0.37 (이전 0.335 / 0.38 / 0.385), DeepSeek 0.315 / 0.315 / 0.31, Haiku 0.24 / 0.245 / 0.25. DeepSeek·Haiku는 이전 실험과 6개 모두 같다(재현 점검 통과).

validation TVD, β* (`out/validation_tvd_beta_star.csv`; 괄호는 이전 k≈72 실험)

| 단계 | EXAONE | DeepSeek | Haiku |
|---|---:|---:|---:|
| c0 | 0.298 (0.298) | 0.282 | 0.300 |
| L0 | 0.239 (0.243) | 0.261 | 0.293 |
| L1 | 0.246 (0.248) | 0.285 | 0.285 |

EXAONE 변화량: c0 +0.0004, L0 −0.0040, L1 −0.0018. 모두 0.01 미만이다. DeepSeek·Haiku 값은 이전과 소수 여섯째 자리까지 같다.

β=1(`out/validation_tvd_beta1.csv`)이면 EXAONE L0 0.406, L1 0.375로 균등 기준선(L0 0.335, L1 0.353)보다 나쁘다. DeepSeek·Haiku도 마찬가지다(L1 0.443, 0.486).

EXAONE k는 프롬프트마다 95~99(중앙 98)이고 DeepSeek·Haiku는 100이다(`kosmos_build.json`). m1(태우)이 진행 중이라 최종값이 아니다.

## 결론

EXAONE 표본을 k≈72에서 k≈98로 늘려도 보정 후 TVD는 0.01 미만으로만 움직였다(표본 수 영향 작음). β*는 세 단계 모두 0.015~0.025 작아졌다. 보정 후 L0·L1에서 EXAONE이 가장 낮은 순서는 그대로다. 단 EXAONE 표본이 DeepSeek·Haiku와 아직 완전히 같지 않다.

## 다음

- m1(태우)이 끝나 EXAONE k=100이 되면 `build_dataset.py`를 다시 돌리고 새 실험으로 다시 잰다.
- 팀이 정하면 test 49문항을 한 번만 연다.
