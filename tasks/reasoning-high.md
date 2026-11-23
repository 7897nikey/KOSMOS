# 추론 high (DeepSeek Eco, 60문항 층화)

## 수집
| 항목 | 값 |
|---|---|
| 확인 인자 | `--reasoning high --items research/design_v2_20261112/subsets/high60.csv --k 50 --dry-run` → `raw/v2/deepseek_eco_high_high60_p1.jsonl`, 36,000건(프롬프트 720개) |
| 스모크 인자 | `--reasoning high --smoke 20` |
| 본 수집 인자 | `--reasoning high --items research/design_v2_20261112/subsets/high60.csv --k 50 --max-usd 8` |
| 예상 | $3~5. 스모크에서 출력이 평균 1,500토큰을 넘으면 멈추고 보고 |

60문항(`subsets/high60.csv`)은 test를 뺀 197문항에서 분할×문항 유형 비례로 뽑았다(calibration 45, validation 15). 만든 코드는 `subsets/build_high60.py`.

## 분석: `experiments/<날짜>-reasoning-high`

**질문**: 추론을 high까지 올리면 셀끼리 차이와 사람 방향 정렬이 계속 커지는가, 멈추는가.

**데이터**: 같은 60문항에서 끔·low(기존 파일), high(새 파일). 추론 medium 파일(`deepseek_eco_medium_p1.jsonl`)이 있으면 같이 넣는다.

**주 지표 2개** (Holm 보정, high − low, 60문항 중 순서형·이진 문항, 문항 부트스트랩 2,000회)
1. 셀 간 분산비 (모델/사람, 잡음 뺀 값)
2. 셀 편차 정렬 (가중 상관)

정의는 `experiments/20261123-reasoning-explain/run.py`를 그대로 쓴다(β 없이 원응답 비율). 문항 선택만 60문항으로 바꾼다.

**탐색** (보정 없음):
- 보정 후 사람 TVD. β는 60문항 중 calibration 45문항에서 고르고 validation 15문항에서 잰다. 문항이 적어 탐색으로만 적는다.
- 기울기, 이동 분해, 무효율, 평균 추론 토큰.
- 끔→low→(medium)→high 추세 표.

**판정**: Holm p < 0.05면 "있음". 방향 예측은 두지 않는다.
