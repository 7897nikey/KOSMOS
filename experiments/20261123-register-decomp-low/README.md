# 20261123-register-decomp-low

| 항목 | 내용 |
|---|---|
| 담당 | joohee |
| 시작 | 2026-11-23 |
| 상태·결론 | `meta.json` (`status`: 진행/완료/중단, `conclusion`: 한 줄) |
| 데이터 | `research/design_v2_20261112/dataset/kosmos_long.csv.gz` (해시는 `out/run.json`) |

## 질문 (돌리기 전에 쓴다)

`20261123-register-final`에서 원문(당신·하십시오체) → '이 사람·하십시오체'는 페르소나 문장, 문항 호칭, 지시문, 주체 높임(-시-) 네 가지가 한꺼번에 바뀌었다. 추론 low(DeepSeek Eco, 한국 성인)에서 이 네 부품 중 보정 후 사람 TVD를 바꾸는 것은 무엇인가?

## 가설과 판정 기준 (돌리기 전에 쓴다)

- 가설: 방향 예측은 두지 않는다.
- 데이터: 원문 = `deepseek_eco_low_p1`의 한국 성인(`level == "L0"`), 이 사람·하십시오체 = `deepseek_eco_low_register_p1`의 `probe_id`가 `__C`로 끝나는 줄, 신규 8조건 중 7조건 = `deepseek_eco_low_decomp_p1`(k=50, 프롬프트 1,968개).
- β: `experiments/20261123-register-final/out/betas.csv`의 한국 성인 β를 10조건에 똑같이 쓴다(low 0.365).
- 주 가설군 4개 (Holm 보정, 보정 후 사람 TVD, validation 49문항, 문항 부트스트랩 2,000회)
  1. 페르소나 문장 효과 = 2×2×2 설계의 '이 사람' 4칸 평균 − '당신' 4칸 평균
  2. 문항 호칭 효과 = 호칭 '이 사람' 4칸 평균 − '귀하' 4칸 평균
  3. 지시문 효과 = '이 사람이 고를' 4칸 평균 − 원문 지시문 4칸 평균
  4. 주체 높임 제거 효과 = 이 사람·하십시오체 − 주체 높임을 유지한 이 사람 전체 적용(`all_keep_si`)
- 판정: Holm 보정 p < 0.05이면 "유의". 구간은 문항 짝 부트스트랩 95%.
- 탐색(보정 없음): 호칭 귀하→당신(`dangsin_address` − 원문), 2×2×2 2차 상호작용 3개, 같은 효과를 197문항에서, 주체 높임 효과를 -시-가 있던 151문항만으로, 회귀 점검(이 사람·하십시오체 − 원문이 register-final의 단순 대비 low −0.020, validation과 같은지), 조건별 무효율, 최빈 보기가 원문과 달라진 비율.
- 쓰는 문항 분할: calibration(β는 기존 값 사용) / validation(확인). test는 쓰지 않는다.
- 분석 설계 출처: `tasks/register-decomp-common.md`, `tasks/register-decomp-low.md`. 조건 정의는 `research/design_v2_20261112/register_decomp/out/examples.md`.

## 방법

`run.py`가 위 설계대로 계산한다. 수집은 `run_deepseek_eco.py --reasoning low --prompts decomp --k 50 --max-usd 5`.

## 결과 (돌린 뒤 쓴다)

`out/`의 표에서 옮긴 숫자만 쓴다. 표 파일 이름을 같이 적는다.

## 결론

한 줄. 가설이 기각돼도 그대로 쓴다.

## 다음
