# register_v1 — 인칭·화계 프롬프트 변환 초안

2×2 설계(추론/비추론 × 인칭/화계)의 인칭·화계 축에 쓸 프롬프트 초안이다. **검수 전 초안**이라 파일 이름에 `_draft`가 붙어 있다.

## 조건

| 페르소나 | A 당신·하십시오체 | B 당신·해라체 | C 이 사람·하십시오체 | D 이 사람·해라체 |
|---|---|---|---|---|
| c0 | 기존 데이터 | 신규 | — | — |
| L0 | 기존 데이터 | 신규 | 신규 | 신규 |

- c0에는 대상 소개 문장이 없어서 "이 사람"이 가리킬 대상이 없다. 그래서 c0는 화계만 바꾼다.
- A는 `prompts_v2.csv`와 글자 단위로 같아야 한다. 빌드 스크립트가 이를 확인하고, 다르면 멈춘다.
- 246문항 전체, 프롬프트 1,476개(신규 984개)다.

## 바꾸는 곳

system, 페르소나 문장(L0), 문항 도입부("..." 앞), 마지막 지시문만 바꾼다. 진술문("..." 뒤)과 선택지는 그대로 둔다. 선택지에는 존댓말이 없다. 진술문은 EMPLY 하나만 해라체에서 고친다.

## 실행

```bash
PYTHONUTF8=1 python research/design_v2_20261112/register_v1/build_register_prompts.py
```

규칙 변환 → `corrections.csv`로 보정 → 자동 점검 → 조립 → `out/review_table.xlsx` 순서다. 같은 입력이면 결과가 같다.

## 파일

| 파일 | 내용 |
|---|---|
| `build_register_prompts.py` | 변환·점검·조립·대조표 |
| `corrections.csv` | 규칙이 못 한 곳을 고친 기록(근거 포함) |
| `out/review_table.xlsx` | 검수용 대조표 (도입부 93종 × B·C·D) |
| `out/prompts_register_v1_draft.csv` | 조립된 프롬프트 |
| `out/lead_table.csv`, `out/summary.json` | 도입부별 규칙안·최종안·점검 결과, 요약 수치 |

## 검수 후

검수 의견을 `corrections.csv`에 반영하고 스크립트를 다시 돌린다. 확정하면 `_draft`를 떼고 해시를 기록한다. 설계 고정 파일(`prompts_v2.csv` 등)은 수정하지 않는다.
