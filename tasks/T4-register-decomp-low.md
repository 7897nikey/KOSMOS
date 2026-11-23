# T4. 인칭 요소 분해, 추론 low (DeepSeek Eco)

조건·분석은 [`_decomp_common.md`](_decomp_common.md)를 따른다.

| 항목 | 값 |
|---|---|
| 확인 인자 | `--reasoning low --prompts decomp --k 50 --dry-run` → `raw/v2/deepseek_eco_low_decomp_p1.jsonl`, 98,400건 |
| 스모크 인자 | `--reasoning low --prompts decomp --smoke 20` |
| 본 수집 인자 | `--reasoning low --prompts decomp --k 50 --max-usd 5` |
| 예상 | $2.1~2.7, 약 5시간 |
| 분석 폴더 | `experiments/<날짜>-register-decomp-low` (원문 `deepseek_eco_low_p1`, 이 사람·하십시오체 `deepseek_eco_low_register_p1`) |

T3 분석 폴더가 먼저 올라왔으면 `run.py`를 복사해 데이터 경로만 바꿔도 된다(출처 주석).
