# 20261123-reasoning-high (추론 high, 60문항 층화)

| 항목 | 내용 |
|---|---|
| 담당 | 태우 |
| 시작 | 2026-11-23 |
| 상태·결론 | `meta.json` |
| 데이터 | 추론 끔 `raw/v2/deepseek_eco_p1.jsonl`, low `raw/v2/deepseek_eco_low_p1.jsonl`, high `raw/v2/deepseek_eco_high_high60_p1.jsonl`(신규, DeepSeek V4.1 Flash Eco, k=50). 추론 medium `raw/v2/deepseek_eco_medium_p1.jsonl`이 있으면 같이 넣는다. 페르소나 없음·한국 성인·성별×연령 10셀, 원문 문체. 정답은 `counts_k100.npz`의 H·NEFF. 해시는 `out/run.json` |
| 문항 | `research/design_v2_20261112/subsets/high60.csv` 60문항. test를 뺀 197문항에서 분할(calibration 45, validation 15) × 문항 유형 비례로 뽑았다(`subsets/build_high60.py`, 시드 20261123) |

## 질문 (돌리기 전에 쓴다)

`20261123-reasoning-explain`에서 추론 끔 → low는 셀끼리 차이를 키우고(분산비 3.38 → 4.51), 그 차이를 사람 셀 차이와 더 같은 방향으로 맞췄다(정렬 0.479 → 0.548). 추론을 high까지 올리면 이 두 가지가 계속 커지는가, 멈추는가.

## 가설과 판정 기준 (돌리기 전에 쓴다)

방향 예측은 두지 않는다. test 49문항은 쓰지 않는다(60문항에 없다).

대상은 60문항 중 순서형·이진 문항(`scale_type`이 ordinal_monotone 또는 binary)이다. 모델 분포는 원응답 비율이고 β를 적용하지 않는다.

**주 지표 2개** (Holm 보정, high − low, 문항 부트스트랩 2,000회, p = 2 × min(비율(≤0), 비율(≥0)), 최소 1/2,000)

| 지표 | 정의 |
|---|---|
| 셀 간 분산비 | Σ문항 V모델 / Σ문항 V사람. V = Σ셀 w(s셀 − s̄)²에서 표집 잡음 기댓값 Σ w(1−w)·분산/표본수를 뺀 값. 10셀 가중치 w는 `cells.csv`의 pop_share |
| 셀 편차 정렬 | 문항·셀을 모두 모은 가중 상관 Σ w·d모델·d사람 / √(Σ w·d모델² · Σ w·d사람²), d = s셀 − s̄ |

정의는 `experiments/20261123-reasoning-explain/run.py`와 같다. 문항 선택만 60문항으로 바꾼다.

판정: Holm 보정 p < 0.05면 "있음"이다.

**탐색** (보정 없음, "탐색"으로 표기)
- 보정 후 사람 TVD. β는 설정마다 페르소나 없음 하나, 한국 성인과 10셀 하나(`experiments/README.md` 통계 정책)이고, 60문항 중 calibration 45문항에서 고른 뒤 validation 15문항에서 잰다. 문항이 적어 탐색으로만 적는다.
- 기울기 Σ w·d모델·d사람 / Σ w·d사람².
- 이동 분해(low → high): 공통 이동과 셀 고유 이동의 제곱합.
- 무효율, 평균 추론 토큰, 길이 초과(`finish_reason == "length"`) 비율.
- 끔 → low → (medium) → high 추세 표. medium은 파일이 있을 때만 넣는다.

## 방법

`run.py` 하나다. 개수 불러오기와 지표 함수는 `20261123-reasoning-explain/run.py`, TVD·β는 `20261118-reasoning-effect/run.py`에서 복사했다(출처 주석).

출력:
- `out/primary.csv`(주 지표와 Holm), `out/explore.csv`
- `out/trend.csv`(설정별 지표), `out/quality.csv`(무효율·추론 토큰)
- `out/run.log`, `out/run.json`

## 결과 (돌린 뒤 쓴다)

## 결론

## 다음
