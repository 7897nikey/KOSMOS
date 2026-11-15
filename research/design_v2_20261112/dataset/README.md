# KOSMOS 분석용 DataFrame

응답 원본(jsonl 수십만 줄)을 **한 장의 long 표**로 줄인 것이다. 이 표와 `silicon.py`(pandas·numpy만)면 보정·평가 실험이 CPU로 수 초 안에 돈다. 원본·키·GPU가 필요 없다.

| 파일 | 내용 |
|---|---|
| `kosmos_long.csv.gz` | 모델 × 페르소나 × 문항 × 보기 한 줄 (약 39,000행, 0.4MB) |
| `silicon.py` | `load` · `calibrate` · `score` · `fit_beta` · `summary` (약 90줄) |
| `example.py` | 바로 돌려 볼 예 |
| `build_dataset.py` | 원응답 → 위 표. EXAONE 결과가 늘면 다시 실행 (`kosmos_build.json`에 소스·k 기록) |

## 열

`model`(exaone/deepseek/haiku) · `level`(c0/L0/L1) · `persona`(c0, L0, L1 셀 id) · `var`(문항) · `code`(보기 값) · `opt_idx` · `n_options` · **`count`**(그 보기를 고른 유효 응답 수) · `n_valid` · `n_total`(k) · `n_refusal` · **`human_p`**(KGSS 2023 FINALWT 가중 분포) · `human_neff`(Kish 유효 표본) · `split`(calibration/validation/test) · `scale_type` · `sensitivity_level` · `topic`

## 보정

`p_k = (count_k + 1)^β / Σ_j (count_j + 1)^β` — 라플라스 평활 후 지수 β. β<1이면 쏠린 분포를 편다. β는 calibration 148문항의 평균 TVD가 최소인 값으로 고른다. 수식 전체는 `../FORMULAS.md`.

## 규칙

- `load()`는 **test 49문항을 뺀다.** 사전등록상 EXAONE k=100이 모인 뒤 한 번만 연다. 열려면 `load(include_test=True)`.
- 프롬프트마다 k가 다를 수 있다(EXAONE은 69~73, 나머지 100). 표본 수를 맞춘 비교는 `../analysis/exaone_interim_v2.py` 결과(`out_exaone_interim/`)가 기준이다. 이 표로 바로 비교하면 EXAONE만 표본이 적어 불리하다.
- 사람 분포는 KGSS 가중 집계값만 들어 있다. 개인 단위 원자료는 없다.

## 검증

DeepSeek·Haiku(k=100)로 기존 `frozen_beta.json`과 대조했다: β* 6개 값(c0·L0·L1×2모델)이 모두 일치, test L0 TVD 0.228/0.271, 균등 0.360도 `.harness/context.md` 기록과 일치.
