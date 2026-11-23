# 추론 medium (DeepSeek Eco, 246문항)

## 수집
| 항목 | 값 |
|---|---|
| 확인 인자 | `--reasoning medium --k 50 --dry-run` → `raw/v2/deepseek_eco_medium_p1.jsonl`, 147,600건 |
| 스모크 인자 | `--reasoning medium --smoke 20` |
| 본 수집 인자 | `--reasoning medium --k 50 --max-usd 12` |
| 예상 | $4~8, 10시간 이상 (low 실측 6.8시간) |

## 분석: `experiments/<날짜>-reasoning-medium`

**질문**: 추론을 low에서 medium으로 올리면 (1) 사람 분포와의 거리, (2) 셀끼리 차이의 크기, (3) 그 차이가 사람 셀 차이와 같은 방향인지가 어떻게 바뀌는가.

**데이터**: 추론 끔 `raw/v2/deepseek_eco_p1.jsonl`, low `raw/v2/deepseek_eco_low_p1.jsonl`, medium(새 파일). 페르소나 없음·한국 성인·성별×연령 10셀. 정답 `counts_k100.npz`의 H·NEFF.

**주 지표 4개** (Holm 보정, medium − low, 문항 부트스트랩 2,000회, p = 2·min(비율≤0, 비율≥0))
1. 보정 후 사람 TVD, 한국 성인 (validation 49문항)
2. 보정 후 사람 TVD, 10셀 pop_share 가중 (validation 49문항)
3. 셀 간 분산비 (모델/사람, 잡음 뺀 값, test 뺀 순서형·이진 문항)
4. 셀 편차 정렬 (가중 상관, 같은 문항)

- β: 설정(끔/low/medium)마다 페르소나 없음 하나, 한국 성인과 10셀이 하나. calibration 148문항에서 TVD 최소, 격자 0.05~1.5(0.005 간격).
- 지표 1·2는 `experiments/20261118-reasoning-effect/run.py`, 지표 3·4는 `experiments/20261123-reasoning-explain/run.py`의 정의를 그대로 쓴다. 함수는 복사하고 출처를 주석으로 남긴다.

**탐색** (보정 없음): medium − 끔(같은 4지표), 기울기, 이동 분해(공통 이동/셀 고유 이동), 무효율, 평균 추론 토큰, 끔→low→medium 추세 표.

**판정**: Holm p < 0.05면 "있음". 방향 예측은 두지 않는다.
