# 담당자별 에이전트 프롬프트

최신 `exp/haiku-pilot-recovery`를 받은 사람이면 아래 준비 두 줄을 실행한 뒤, 맡은 프롬프트를 Claude Code나 Codex에 그대로 붙여 넣으면 된다. 담당은 아직 정하지 않았다.

| 담당 | 수집 조건 | 비용 | 수집 시간 | 추가로 맡을 일 |
|---|---|---|---|---|
| 추론 medium | DeepSeek Eco, 추론 medium, 원문, 페르소나 없음·한국 성인·성별×연령 10셀, 246문항 | $4~8 | 10시간 이상 | 없음 |
| 추론 high | DeepSeek Eco, 추론 high, 원문, 페르소나 없음·한국 성인·성별×연령 10셀, 60문항 | $3~5 | 수 시간 | 추론 강도 추세표(끔→low→medium→high) |
| 요소 분해, 추론 끔 | DeepSeek Eco, 추론 끔, 인칭 요소 분해 8조건, 한국 성인, 246문항 | $0.4 | 짧음 | 분해 분석 코드 먼저 공개, 다교님 계획서와 겹치는 조건 정리 |
| 요소 분해, 추론 low | DeepSeek Eco, 추론 low, 인칭 요소 분해 8조건, 한국 성인, 246문항 | $2.1~2.7 | 약 5시간 | 끔·low 통합 정리 |

---

## 1. 추론 medium

**붙여 넣기 전에 사람이 직접 할 일** (키는 에이전트에게 보여 주지 않는다)

```bash
cd KOSMOS && git fetch upstream && git switch -c exp/reasoning-medium-<내이름> upstream/exp/haiku-pilot-recovery
printf 'OPENGATEWAY_API_KEY=%s\n' '<내 키>' > .env.opengateway
```

```text
너는 KOSMOS 레포 루트에서 "추론 medium (DeepSeek Eco, 페르소나 없음·한국 성인·성별×연령 10셀, 246문항, k=50)" 실험을 수집부터 분석·PR 초안까지 진행한다. 한국어로 보고한다.

[먼저 읽을 파일]
   - `AGENTS.md`
   - `experiments/README.md` (실험 폴더 순서와 통계 정책)
   - `tasks/README.md` (수집·분석 공통 절차)
   - `tasks/reasoning-medium.md`

[승인 범위]
- 이 프롬프트가 이 실험의 유료 API 호출 승인이다. 단 누적 추정 비용 $12 안에서만 유효하다. 수집 명령에는 항상 `--max-usd 12`를 붙인다.
- 다른 실험의 수집, 이미 있는 원응답·설계 고정 파일 수정, test 49문항 분석은 하지 않는다.

[키]
- 키는 `.env.opengateway`에 이미 있다. 파일이 있는지만 확인하고 내용을 출력·인용·로그에 남기지 않는다.
- 수집 명령은 같은 셸 줄에서 `set -a; . ./.env.opengateway; set +a; python3 ...` 형태로 실행한다.

[순서]
1. 환경 확인
   - 현재 브랜치가 `upstream/exp/haiku-pilot-recovery`에서 갈라졌고 `tasks/PROMPTS.md`가 있는지 확인한다. 없으면 멈추고 보고한다.
   - `raw/v2/deepseek_eco_p1.jsonl`이 없으면 `tasks/README.md` 0단계 3)의 명령으로 원응답을 푼다.
2. 수집 확인: 테스크 파일의 "확인 인자"로 `--dry-run`을 돌려 출력 파일 이름과 호출 수가 테스크 파일과 같은지 확인한다.
3. 스모크: 테스크 파일의 "스모크 인자"에 `--max-usd 0.5`를 붙여 돌린다.
   - 유효 응답률 95% 미만, HTTP 400, 또는 스모크 비용으로 환산한 본 수집 비용이 $12를 넘으면 멈추고 숫자와 함께 보고한다.
4. 세팅 커밋 (결과를 보기 전): `experiments/new.py`로 `experiments/<오늘 날짜>-reasoning-medium` 폴더를 만들고, README의 질문·가설·판정 기준 칸에 테스크 파일의 분석 설계를 옮겨 적는다. `exp: <ID> 세팅`으로 커밋하고 `git push -u origin HEAD`.
5. 본 수집: 테스크 파일의 "본 수집 인자"로 `nohup ... > /dev/null 2>&1 &` 백그라운드 실행한다.
   - 10시간 이상 걸린다. 세션이 끝나도 수집은 계속 돈다. 시작 직후 나에게 확인·재개 명령을 알려 준다.
   - 진행은 `raw/v2/<출력 이름>.log` 마지막 줄로 본다. 끝 줄이 `완료`가 아니면(끊김, `api_error`, 예산 멈춤) 같은 명령을 다시 실행한다. 예산 상한에서 멈췄는데 남은 호출이 있으면 더 돌리지 말고 보고한다.
   - 기다리는 동안 `run.py`를 작성한다. 쌓이는 중인 부분 데이터로 코드가 도는지만 확인한다(결론은 내지 않는다).
6. 원응답 업로드: `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<출력 이름>.jsonl` → `results/README.md` 표에 한 줄 추가 → 커밋.
7. 분석: `run.py`를 커밋한 뒤 실행한다(`out/run.json`의 `git_dirty`가 false여야 한다). README 결과·결론과 `meta.json`의 status·conclusion을 채우고 `python experiments/index.py && python experiments/index.py --check`를 통과시킨다. `exp: <ID> <결론 한 줄>`로 커밋하고 push한다.
8. 추가 일 없음. 단, 추론 high 담당이 추세표를 만들 수 있도록 수집이 끝나면 원응답 압축본을 먼저 push하고 보고에 파일 이름을 적는다.
9. PR 초안: 중앙 레포(Great-Grace/KOSMOS)로 보낼 PR 제목·본문을 써서 보여 준다. PR은 내가 확인한 뒤에 연다.

[보고 규칙]
- 숫자는 `out/*.csv`와 수집 로그에서 그대로 옮긴다. 기억이나 추정으로 쓰지 않는다.
- 조건 이름은 풀어 쓴다(A/B/C/D, c0/L0 같은 약어 금지).
- 마지막 보고: 실제 사용 비용, 유효 응답률, 주 지표 결과(차이·95% 구간·Holm p), 결론 한 줄, 남은 문제.
```

---

## 2. 추론 high

**붙여 넣기 전에 사람이 직접 할 일** (키는 에이전트에게 보여 주지 않는다)

```bash
cd KOSMOS && git fetch upstream && git switch -c exp/reasoning-high-<내이름> upstream/exp/haiku-pilot-recovery
printf 'OPENGATEWAY_API_KEY=%s\n' '<내 키>' > .env.opengateway
```

```text
너는 KOSMOS 레포 루트에서 "추론 high (DeepSeek Eco, 페르소나 없음·한국 성인·성별×연령 10셀, 60문항 층화, k=50)" 실험을 수집부터 분석·PR 초안까지 진행한다. 한국어로 보고한다.

[먼저 읽을 파일]
   - `AGENTS.md`
   - `experiments/README.md` (실험 폴더 순서와 통계 정책)
   - `tasks/README.md` (수집·분석 공통 절차)
   - `tasks/reasoning-high.md`

[승인 범위]
- 이 프롬프트가 이 실험의 유료 API 호출 승인이다. 단 누적 추정 비용 $8 안에서만 유효하다. 수집 명령에는 항상 `--max-usd 8`를 붙인다.
- 다른 실험의 수집, 이미 있는 원응답·설계 고정 파일 수정, test 49문항 분석은 하지 않는다.

[키]
- 키는 `.env.opengateway`에 이미 있다. 파일이 있는지만 확인하고 내용을 출력·인용·로그에 남기지 않는다.
- 수집 명령은 같은 셸 줄에서 `set -a; . ./.env.opengateway; set +a; python3 ...` 형태로 실행한다.

[순서]
1. 환경 확인
   - 현재 브랜치가 `upstream/exp/haiku-pilot-recovery`에서 갈라졌고 `tasks/PROMPTS.md`가 있는지 확인한다. 없으면 멈추고 보고한다.
   - `raw/v2/deepseek_eco_p1.jsonl`이 없으면 `tasks/README.md` 0단계 3)의 명령으로 원응답을 푼다.
2. 수집 확인: 테스크 파일의 "확인 인자"로 `--dry-run`을 돌려 출력 파일 이름과 호출 수가 테스크 파일과 같은지 확인한다.
3. 스모크: 테스크 파일의 "스모크 인자"에 `--max-usd 0.5`를 붙여 돌린다.
   - 유효 응답률 95% 미만, HTTP 400, 또는 스모크 비용으로 환산한 본 수집 비용이 $8를 넘으면 멈추고 숫자와 함께 보고한다.
4. 세팅 커밋 (결과를 보기 전): `experiments/new.py`로 `experiments/<오늘 날짜>-reasoning-high` 폴더를 만들고, README의 질문·가설·판정 기준 칸에 테스크 파일의 분석 설계를 옮겨 적는다. `exp: <ID> 세팅`으로 커밋하고 `git push -u origin HEAD`.
5. 본 수집: 테스크 파일의 "본 수집 인자"로 `nohup ... > /dev/null 2>&1 &` 백그라운드 실행한다.
   - 스모크에서 출력이 평균 1,500토큰을 넘으면 본 수집 전에 멈추고 보고한다.
   - 진행은 `raw/v2/<출력 이름>.log` 마지막 줄로 본다. 끝 줄이 `완료`가 아니면(끊김, `api_error`, 예산 멈춤) 같은 명령을 다시 실행한다. 예산 상한에서 멈췄는데 남은 호출이 있으면 더 돌리지 말고 보고한다.
   - 기다리는 동안 `run.py`를 작성한다. 쌓이는 중인 부분 데이터로 코드가 도는지만 확인한다(결론은 내지 않는다).
6. 원응답 업로드: `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<출력 이름>.jsonl` → `results/README.md` 표에 한 줄 추가 → 커밋.
7. 분석: `run.py`를 커밋한 뒤 실행한다(`out/run.json`의 `git_dirty`가 false여야 한다). README 결과·결론과 `meta.json`의 status·conclusion을 채우고 `python experiments/index.py && python experiments/index.py --check`를 통과시킨다. `exp: <ID> <결론 한 줄>`로 커밋하고 push한다.
8. 추가 일: 추론 강도 추세표
   - `git fetch upstream origin`과 중앙 레포 PR·브랜치에서 `results/raw_gz/v2/deepseek_eco_medium_p1.jsonl*.gz`가 올라왔는지 확인한다.
   - 있으면 풀어서 같은 60문항으로 끔→low→medium→high 추세표(주 지표 2개, 보정 후 사람 TVD, 무효율, 평균 추론 토큰)를 `out/trend.csv`와 README 표로 추가하고 다시 커밋한다.
   - 아직 없으면 끔→low→high로 마무리하고, medium이 올라오면 다시 돌릴 수 있게 `run.py`가 파일 유무를 보고 자동으로 포함하도록 짠다.
9. PR 초안: 중앙 레포(Great-Grace/KOSMOS)로 보낼 PR 제목·본문을 써서 보여 준다. PR은 내가 확인한 뒤에 연다.

[보고 규칙]
- 숫자는 `out/*.csv`와 수집 로그에서 그대로 옮긴다. 기억이나 추정으로 쓰지 않는다.
- 조건 이름은 풀어 쓴다(A/B/C/D, c0/L0 같은 약어 금지).
- 마지막 보고: 실제 사용 비용, 유효 응답률, 주 지표 결과(차이·95% 구간·Holm p), 결론 한 줄, 남은 문제.
```

---

## 3. 요소 분해, 추론 끔

**붙여 넣기 전에 사람이 직접 할 일** (키는 에이전트에게 보여 주지 않는다)

```bash
cd KOSMOS && git fetch upstream && git switch -c exp/register-decomp-off-<내이름> upstream/exp/haiku-pilot-recovery
printf 'OPENGATEWAY_API_KEY=%s\n' '<내 키>' > .env.opengateway
```

```text
너는 KOSMOS 레포 루트에서 "인칭 요소 분해, 추론 끔 (DeepSeek Eco, 한국 성인, 8조건 × 246문항, k=50)" 실험을 수집부터 분석·PR 초안까지 진행한다. 한국어로 보고한다.

[먼저 읽을 파일]
   - `AGENTS.md`
   - `experiments/README.md` (실험 폴더 순서와 통계 정책)
   - `tasks/README.md` (수집·분석 공통 절차)
   - `tasks/register-decomp-off.md`
   - `tasks/register-decomp-common.md`

[승인 범위]
- 이 프롬프트가 이 실험의 유료 API 호출 승인이다. 단 누적 추정 비용 $2 안에서만 유효하다. 수집 명령에는 항상 `--max-usd 2`를 붙인다.
- 다른 실험의 수집, 이미 있는 원응답·설계 고정 파일 수정, test 49문항 분석은 하지 않는다.

[키]
- 키는 `.env.opengateway`에 이미 있다. 파일이 있는지만 확인하고 내용을 출력·인용·로그에 남기지 않는다.
- 수집 명령은 같은 셸 줄에서 `set -a; . ./.env.opengateway; set +a; python3 ...` 형태로 실행한다.

[순서]
1. 환경 확인
   - 현재 브랜치가 `upstream/exp/haiku-pilot-recovery`에서 갈라졌고 `tasks/PROMPTS.md`가 있는지 확인한다. 없으면 멈추고 보고한다.
   - `raw/v2/deepseek_eco_p1.jsonl`이 없으면 `tasks/README.md` 0단계 3)의 명령으로 원응답을 푼다.
2. 수집 확인: 테스크 파일의 "확인 인자"로 `--dry-run`을 돌려 출력 파일 이름과 호출 수가 테스크 파일과 같은지 확인한다.
3. 스모크: 테스크 파일의 "스모크 인자"에 `--max-usd 0.5`를 붙여 돌린다.
   - 유효 응답률 95% 미만, HTTP 400, 또는 스모크 비용으로 환산한 본 수집 비용이 $2를 넘으면 멈추고 숫자와 함께 보고한다.
4. 세팅 커밋 (결과를 보기 전): `experiments/new.py`로 `experiments/<오늘 날짜>-register-decomp-off` 폴더를 만들고, README의 질문·가설·판정 기준 칸에 테스크 파일의 분석 설계를 옮겨 적는다. `exp: <ID> 세팅`으로 커밋하고 `git push -u origin HEAD`.
5. 본 수집: 테스크 파일의 "본 수집 인자"로 `nohup ... > /dev/null 2>&1 &` 백그라운드 실행한다.
   - 출력이 2토큰이라 빨리 끝난다.
   - 진행은 `raw/v2/<출력 이름>.log` 마지막 줄로 본다. 끝 줄이 `완료`가 아니면(끊김, `api_error`, 예산 멈춤) 같은 명령을 다시 실행한다. 예산 상한에서 멈췄는데 남은 호출이 있으면 더 돌리지 말고 보고한다.
   - 기다리는 동안 `run.py`를 작성한다. 쌓이는 중인 부분 데이터로 코드가 도는지만 확인한다(결론은 내지 않는다).
6. 원응답 업로드: `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<출력 이름>.jsonl` → `results/README.md` 표에 한 줄 추가 → 커밋.
7. 분석: `run.py`를 커밋한 뒤 실행한다(`out/run.json`의 `git_dirty`가 false여야 한다). README 결과·결론과 `meta.json`의 status·conclusion을 채우고 `python experiments/index.py && python experiments/index.py --check`를 통과시킨다. `exp: <ID> <결론 한 줄>`로 커밋하고 push한다.
8. 추가 일
   - 분석 코드 먼저 공개: 7단계가 끝나는 대로 push하고, 추론 low 담당이 복사해 쓸 수 있게 `run.py` 맨 위에 데이터 경로를 바꾸는 한 줄(예: `SETTING = "끔"`)만 바꾸면 되도록 짠다.
   - 다교님 계획서와 겹치는 조건 정리: 다교님 계획서(슬랙 #26-2_nlp에 올라온 인칭·화계 기제 1단계 README)는 레포에 없다. 내가 그 내용을 붙여 주면 그 조건들과 이 8조건을 대조한 표(같음/비슷함/없음)를 README '다음' 칸에 적는다. 붙여 주지 않았으면 이 일은 건너뛰고 보고에 '계획서 미제공'이라고 적는다. 슬랙에는 아무것도 올리지 않는다.
9. PR 초안: 중앙 레포(Great-Grace/KOSMOS)로 보낼 PR 제목·본문을 써서 보여 준다. PR은 내가 확인한 뒤에 연다.

[보고 규칙]
- 숫자는 `out/*.csv`와 수집 로그에서 그대로 옮긴다. 기억이나 추정으로 쓰지 않는다.
- 조건 이름은 풀어 쓴다(A/B/C/D, c0/L0 같은 약어 금지).
- 마지막 보고: 실제 사용 비용, 유효 응답률, 주 지표 결과(차이·95% 구간·Holm p), 결론 한 줄, 남은 문제.
```

---

## 4. 요소 분해, 추론 low

**붙여 넣기 전에 사람이 직접 할 일** (키는 에이전트에게 보여 주지 않는다)

```bash
cd KOSMOS && git fetch upstream && git switch -c exp/register-decomp-low-<내이름> upstream/exp/haiku-pilot-recovery
printf 'OPENGATEWAY_API_KEY=%s\n' '<내 키>' > .env.opengateway
```

```text
너는 KOSMOS 레포 루트에서 "인칭 요소 분해, 추론 low (DeepSeek Eco, 한국 성인, 8조건 × 246문항, k=50)" 실험을 수집부터 분석·PR 초안까지 진행한다. 한국어로 보고한다.

[먼저 읽을 파일]
   - `AGENTS.md`
   - `experiments/README.md` (실험 폴더 순서와 통계 정책)
   - `tasks/README.md` (수집·분석 공통 절차)
   - `tasks/register-decomp-low.md`
   - `tasks/register-decomp-common.md`

[승인 범위]
- 이 프롬프트가 이 실험의 유료 API 호출 승인이다. 단 누적 추정 비용 $5 안에서만 유효하다. 수집 명령에는 항상 `--max-usd 5`를 붙인다.
- 다른 실험의 수집, 이미 있는 원응답·설계 고정 파일 수정, test 49문항 분석은 하지 않는다.

[키]
- 키는 `.env.opengateway`에 이미 있다. 파일이 있는지만 확인하고 내용을 출력·인용·로그에 남기지 않는다.
- 수집 명령은 같은 셸 줄에서 `set -a; . ./.env.opengateway; set +a; python3 ...` 형태로 실행한다.

[순서]
1. 환경 확인
   - 현재 브랜치가 `upstream/exp/haiku-pilot-recovery`에서 갈라졌고 `tasks/PROMPTS.md`가 있는지 확인한다. 없으면 멈추고 보고한다.
   - `raw/v2/deepseek_eco_p1.jsonl`이 없으면 `tasks/README.md` 0단계 3)의 명령으로 원응답을 푼다.
2. 수집 확인: 테스크 파일의 "확인 인자"로 `--dry-run`을 돌려 출력 파일 이름과 호출 수가 테스크 파일과 같은지 확인한다.
3. 스모크: 테스크 파일의 "스모크 인자"에 `--max-usd 0.5`를 붙여 돌린다.
   - 유효 응답률 95% 미만, HTTP 400, 또는 스모크 비용으로 환산한 본 수집 비용이 $5를 넘으면 멈추고 숫자와 함께 보고한다.
4. 세팅 커밋 (결과를 보기 전): `experiments/new.py`로 `experiments/<오늘 날짜>-register-decomp-low` 폴더를 만들고, README의 질문·가설·판정 기준 칸에 테스크 파일의 분석 설계를 옮겨 적는다. `exp: <ID> 세팅`으로 커밋하고 `git push -u origin HEAD`.
5. 본 수집: 테스크 파일의 "본 수집 인자"로 `nohup ... > /dev/null 2>&1 &` 백그라운드 실행한다.
   - 약 5시간 걸린다. 시작 직후 나에게 확인·재개 명령을 알려 준다.
   - 진행은 `raw/v2/<출력 이름>.log` 마지막 줄로 본다. 끝 줄이 `완료`가 아니면(끊김, `api_error`, 예산 멈춤) 같은 명령을 다시 실행한다. 예산 상한에서 멈췄는데 남은 호출이 있으면 더 돌리지 말고 보고한다.
   - 기다리는 동안 `run.py`를 작성한다. 쌓이는 중인 부분 데이터로 코드가 도는지만 확인한다(결론은 내지 않는다).
6. 원응답 업로드: `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<출력 이름>.jsonl` → `results/README.md` 표에 한 줄 추가 → 커밋.
7. 분석: `run.py`를 커밋한 뒤 실행한다(`out/run.json`의 `git_dirty`가 false여야 한다). README 결과·결론과 `meta.json`의 status·conclusion을 채우고 `python experiments/index.py && python experiments/index.py --check`를 통과시킨다. `exp: <ID> <결론 한 줄>`로 커밋하고 push한다.
8. 추가 일: 끔·low 통합 정리
   - 중앙 레포에 `register-decomp-off` 실험이 올라왔으면 그 `run.py`를 복사해 설정만 low로 바꿔 쓴다(출처 주석). 없으면 `tasks/register-decomp-common.md`대로 직접 짠다.
   - 끔 결과가 있으면 README에 두 설정의 주 효과 4개(페르소나 문장, 문항 호칭, 지시문, 주체 높임 제거)를 나란히 놓은 표와 '두 설정 공통/한쪽만' 분류를 추가한다.
9. PR 초안: 중앙 레포(Great-Grace/KOSMOS)로 보낼 PR 제목·본문을 써서 보여 준다. PR은 내가 확인한 뒤에 연다.

[보고 규칙]
- 숫자는 `out/*.csv`와 수집 로그에서 그대로 옮긴다. 기억이나 추정으로 쓰지 않는다.
- 조건 이름은 풀어 쓴다(A/B/C/D, c0/L0 같은 약어 금지).
- 마지막 보고: 실제 사용 비용, 유효 응답률, 주 지표 결과(차이·95% 구간·Holm p), 결론 한 줄, 남은 문제.
```
