# 분담 실험 (2026-11-23 등록)

사람마다 테스크 하나를 맡는다. 자기 OpenGateway 키와 충전한 크레딧으로 돌린다.
**AI(Claude Code, Codex 등)에게는 이 파일과 맡은 테스크 파일을 함께 넘긴다.**
예: "`tasks/README.md`와 `tasks/reasoning-medium.md`를 읽고 그대로 진행해."
사람이 테스크 파일을 넘긴 것이 그 테스크의 유료 호출 승인이다. 단, 테스크 파일의 `--max-usd` 상한 안에서만 유효하다.

| 테스크 파일 | 모델 | 추론 | 인칭/화계 | 페르소나 | 문항 | 호출 수 | 예상 비용 | 상한 `--max-usd` |
|---|---|---|---|---|---|---|---|---|
| [추론 medium](reasoning-medium.md) | DeepSeek Eco | medium | 원문 | 페르소나 없음, 한국 성인, 성별×연령 10셀 | 246 | 147,600 | $4~8 | 12 |
| [추론 high](reasoning-high.md) | DeepSeek Eco | high | 원문 | 페르소나 없음, 한국 성인, 성별×연령 10셀 | 60 (`subsets/high60.csv`) | 36,000 | $3~5 | 8 |
| [요소 분해, 추론 끔](register-decomp-off.md) | DeepSeek Eco | 끔 | 요소 분해 8조건 | 한국 성인 | 246 | 98,400 | $0.4 | 2 |
| [요소 분해, 추론 low](register-decomp-low.md) | DeepSeek Eco | low | 요소 분해 8조건 | 한국 성인 | 246 | 98,400 | $2.1~2.7 | 5 |

반복 수는 모두 k=50이다. 테스크끼리 의존하지 않는다. 다만 추론 high 분석은 추론 medium 결과가 있으면 같이 비교한다.

**담당자별 에이전트 프롬프트는 [`PROMPTS.md`](PROMPTS.md)에 있다.** 복사해서 그대로 붙여 넣는다.

## 0. 처음 한 번 (공통)

```bash
# 1) 받기 — 중앙 레포는 Great-Grace/KOSMOS
gh repo fork Great-Grace/KOSMOS --clone && cd KOSMOS
git fetch upstream
# 2) 기준 브랜치에서 작업 브랜치 만들기 (이 테스크 파일들이 main에 합쳐지기 전까지는 exp/haiku-pilot-recovery가 기준)
git switch -c exp/<테스크 파일 이름>-<이름> upstream/exp/haiku-pilot-recovery
# 3) 기존 원응답 풀기 (분석에 필요)
mkdir -p raw/v2 && for f in results/raw_gz/v2/*.jsonl.gz; do gzip -dc "$f" > "raw/v2/$(basename "$f" .gz)"; done
# 4) 키 넣기 — 파일 이름은 .env.opengateway (gitignore 대상). 키 값은 출력·로그·대화에 남기지 않는다
printf 'OPENGATEWAY_API_KEY=%s\n' '<여기에 본인 키>' > .env.opengateway
set -a; . ./.env.opengateway; set +a
```

- 수집기는 표준 라이브러리만 쓴다. `python3`(3.10 이상)이면 된다.
- 분석은 `uv run -q --no-project --python 3.12 --with pandas --with numpy python <스크립트>`로 돌린다(uv가 없으면 pandas·numpy를 설치한 python).

## 1. 수집 순서 (모든 테스크 공통)

R=`research/design_v2_20261112/runners/run_deepseek_eco.py`, 테스크 파일의 "수집 인자"를 `<인자>`에 넣는다.

1. **확인**: `python3 $R <인자> --dry-run` → 출력 파일 이름과 호출 수가 테스크 표와 같은지 본다. 키 없이도 된다.
2. **스모크**: `python3 $R <스모크 인자> --max-usd 0.5`
   - 유효 응답률이 95% 미만이거나 HTTP 400(설정 거부)이면 **멈추고 사람에게 보고**한다.
   - 로그 마지막 줄의 누적 비용을 스모크 건수로 나눠 본 수집 비용을 다시 계산한다. 예상 비용표의 상한을 넘으면 멈추고 보고한다.
3. **본 수집**: `nohup python3 $R <인자> --max-usd <상한> > /dev/null 2>&1 &`
   - 진행은 `tail -f raw/v2/<출력 이름>.log`로 본다. 1분마다 진행·유효율·누적 비용·남은 시간이 찍힌다.
   - 끊기거나 예산 상한에서 멈추면 **같은 명령을 다시 실행**한다. 이미 받은 응답은 건너뛴다.
   - 끝 줄이 `완료`여야 한다. `api_error N건`이면 같은 명령을 다시 돌린다.
4. **올리기**: `python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<출력 이름>.jsonl`
   - `results/raw_gz/v2/`에 압축본이 생긴다(90MB가 넘으면 part로 나뉜다).
   - `results/README.md`의 표에 한 줄 추가한다(파일, 내용, 줄 수, 쓰는 실험).

`--workers` 기본값은 48이다(키당 동시 한도 64). 429가 계속 나면 32로 줄인다.

## 2. 분석 순서 (모든 테스크 공통)

`experiments/README.md`의 순서를 그대로 따른다.

1. `python3 experiments/new.py <이름> --author <이름> --question "<테스크 파일의 질문>"`
2. README에 테스크 파일의 "질문·가설·판정 기준"을 옮겨 적고 **결과를 보기 전에** 세팅 커밋을 push한다.
3. `run.py`를 쓰고 커밋한 뒤 실행한다(커밋 전에 돌린 결과는 `git_dirty`로 막힌다).
4. README 결과·결론, `meta.json` 상태·결론을 채운다. `python3 experiments/index.py && python3 experiments/index.py --check`.
5. 커밋·push 후 중앙 레포로 PR. PR 제목·본문은 사람이 확인한 뒤 연다.

## 3. 지킬 것

- test 49문항(`split == "test"`)은 분석에 쓰지 않는다. 수집에는 포함돼도 된다.
- 보정 β는 `experiments/README.md`의 "통계 정책"을 따른다.
- 설계 고정 파일(`prompts_v2.csv`, `cells.csv`, `register_v1/out/*`, `register_decomp/out/prompts_decomp.csv`, `subsets/high60.csv`)과 기존 원응답은 고치지 않는다. runner가 해시를 확인한다.
- 다른 실험 폴더의 결과를 고치지 않는다.
- 결과 보고에는 조건 이름을 풀어 쓴다(A/B/C/D 같은 약어 금지).
- 문서에 쓰는 숫자는 `out/*.csv`에서 옮긴다.
