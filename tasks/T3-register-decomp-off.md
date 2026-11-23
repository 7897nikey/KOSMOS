# T3. 인칭 요소 분해, 추론 끔 (DeepSeek Eco)

조건·분석은 [`_decomp_common.md`](_decomp_common.md)를 따른다.

| 항목 | 값 |
|---|---|
| 확인 인자 | `--prompts decomp --k 50 --dry-run` → `raw/v2/deepseek_eco_decomp_p1.jsonl`, 98,400건 |
| 스모크 인자 | `--prompts decomp --smoke 20` |
| 본 수집 인자 | `--prompts decomp --k 50 --max-usd 2` |
| 예상 | 약 $0.4 (출력이 2토큰이라 low보다 훨씬 빠르다) |
| 분석 폴더 | `experiments/<날짜>-register-decomp-off` (원문 `deepseek_eco_p1`, 이 사람·하십시오체 `deepseek_eco_register_p1`) |
