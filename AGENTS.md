# AGENTS.md — 이 레포에서 일하는 AI 에이전트용 안내

사람 팀원은 에이전트에게 "AGENTS.md를 읽고 진행해줘"라고만 하면 된다. 에이전트는 아래 순서를 따른다.

## 이 프로젝트

한국형 실리콘 샘플링 검증. LLM 3종(K-EXAONE 2.5, DeepSeek V4.1 Flash, Claude Haiku 5.5)에게 KGSS 2023의 246문항을 페르소나 없이(c0), "한국 성인"으로(L0), 성별×연령 10셀로(L1) 물었다. 프롬프트마다 약 100번씩 물어서 보기 분포를 만들었고, 그 분포가 실제 한국인 응답 분포(FINALWT 가중)와 얼마나 가까운지 잰다.
기준 설계는 `research/design_v2_20261112/DESIGN.md`, 수식은 `research/design_v2_20261112/FORMULAS.md`에 있다.

## 1. 환경 (처음 한 번)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python -m pytest -q tests/test_silicon.py          # 4 passed면 준비 끝
```

- CPU와 pandas·numpy만 쓴다. API 키·GPU·KGSS 원자료(`.sav`)는 필요 없다.
- 루트의 `requirements.txt`는 v1(GPU) 것이고 UTF-16이다. 분석에는 쓰지 않는다.

## 2. 데이터

`research/design_v2_20261112/dataset/kosmos_long.csv.gz`: 모델 × 페르소나 × 문항 × 보기 한 줄짜리 표다(약 39,000행). 열 설명은 같은 폴더의 `README.md`에 있다.

```python
import sys; sys.path.insert(0, "research/design_v2_20261112/dataset")
import silicon as sl
df = sl.load()                 # test 49문항은 빠진다
beta = sl.fit_beta(df)         # calibration 148문항에서 모델×단계별 β*
sl.summary(df[df.split == "validation"], beta)
```

- 보정식: `p_k ∝ (count_k + 1)^β`
- EXAONE은 아직 프롬프트당 69~73표본이고, 나머지 두 모델은 100표본이다(`kosmos_build.json`). 모델끼리 비교할 때는 이 차이를 결과에 적는다.

## 3. 실험하고 기록하기

반드시 `experiments/README.md`의 순서를 따른다. 요약하면 다음과 같다.

0. 중앙 레포는 `Great-Grace/KOSMOS`다. 사람의 fork를 클론해서 작업한다(`origin` = fork, `upstream` = 중앙). `upstream`이 없으면 `experiments/README.md`의 "처음 한 번: fork"대로 설정한다. push는 fork로만 하고, 중앙에는 PR로만 보낸다.
1. 최신 중앙 기준으로 실험마다 브랜치를 만들고 폴더를 만든다.
   ```bash
   git fetch upstream && git switch -c exp/<날짜>-<영문-이름> upstream/HEAD
   .venv/bin/python experiments/new.py <영문-이름> --author <사람 이름> --question "<한 줄>"
   ```
2. 돌리기 전에 그 폴더의 `README.md`에 질문·가설·판정 기준을 쓰고 커밋·push한다.
3. `run.py`를 고쳐 커밋한 뒤 실행한다. `sl.save_run()`이 `out/`에 표와 `run.json`을 남긴다(커밋, 데이터 해시, 파라미터, test 사용 여부).
4. `out/`의 숫자로 README의 결과·결론을 채운다. `meta.json`의 `status`와 `conclusion`을 고친다.
5. 아래 두 명령을 차례로 돌려 통과시킨다.
   ```bash
   .venv/bin/python experiments/index.py
   .venv/bin/python experiments/index.py --check
   ```
6. fork에 push하고 `gh pr create --repo Great-Grace/KOSMOS --fill`로 중앙에 PR을 연다. PR 템플릿 체크리스트를 채운다. GitHub Actions가 테스트와 `--check`를 다시 돌린다.

- `experiments/LOG.md`는 자동 생성 파일이라 손으로 고치지 않는다. 충돌하면 `index.py`를 다시 돌린다.
- PR을 보내는 것은 사람 이름으로 남는 일이다. PR 제목·본문을 보여 주고 사람이 확인한 뒤에 연다.
- 예시는 `experiments/20261115-baseline-calibration/`이다.

## 4. 지켜야 할 것

- **test 49문항(`split == "test"`)을 열지 않는다.** 사전등록 규칙이다. EXAONE 100표본이 모인 뒤 팀이 정해서 한 번만 연다. `load(include_test=True)`는 사람이 명시적으로 지시할 때만 쓴다.
- **유료 API를 호출하지 않는다.** 응답 수집(`research/design_v2_20261112/runners/`)은 비용이 드니 사람의 승인을 받는다.
- API 키는 사람마다 자기 것을 쓴다. 양식은 `.env.example`이다(`cp .env.example .env` → 키 입력 → `set -a; . ./.env; set +a`). `.env`는 커밋하지 않고, 키 값을 출력·로그·대화에 남기지 않는다. 분석·실험에는 키가 필요 없다.
- 다음은 설계를 고정하는 파일이라 고치지 않는다: `MANIFEST.json`, `prompts_v2.csv`, `item_manifest_v2.csv`, `cells.csv`, `analysis/out_calib/frozen_beta.json`, 원응답(`raw/`, `results/raw_gz/`).
- 문서에 쓰는 숫자는 실행 결과 파일에서 옮긴다. 기억이나 추정으로 쓰지 않는다.
- 기각된 결과도 지우지 않고 기록한다.
- 커밋 메시지는 한국어로 `feat:`/`docs:`/`exp:` 접두사를 붙인다. 실험 결과 커밋은 `exp: <실험 ID> <결론 한 줄>` 형식이다.

## 5. 어디에 무엇이 있나

| 경로 | 내용 |
|---|---|
| `experiments/` | **실험과 기록(여기서 작업)**. `LOG.md`가 전체 목록이다 |
| `research/design_v2_20261112/dataset/` | 분석용 표, `silicon.py`, `build_dataset.py`(원응답 → 표) |
| `research/design_v2_20261112/DESIGN.md`, `FORMULAS.md` | 확정 설계와 수식 |
| `research/design_v2_20261112/analysis/` | 지금까지의 본 분석(노트북, `out_calib/`, `out_exaone_interim/REPORT.md`) |
| `research/design_v2_20261112/runners/`, `team_template/` | API 응답 수집기, K-EXAONE 팀 분담 패키지(유료·키 필요) |
| `results/raw_gz/v2/` | DeepSeek·Haiku 원응답(gzip). EXAONE 원응답은 아직 레포에 없다 |
| `research/preregistration_*.md`, `gates*.md` | 사전등록과 단계별 판정 |
| 루트의 `kgss_*.py`, `audit_outputs/`, `tests/test_*`(silicon 제외) | v1(10월 이전, Colab GPU) 코드와 감사 기록. 참고용 |

## 6. 데이터 표를 다시 만들 때 (원응답이 있는 사람만)

```bash
.venv-nb/bin/python research/design_v2_20261112/dataset/build_dataset.py
```

- KGSS 정제본 `inventory/kgss_clean.parquet`과 pyarrow가 필요하다. 레포에는 없다.
- EXAONE 결과 zip은 `research/design_v2_20261112/team_packages/KOSMOS_EXAONE_m1_태우/`에 둔다.
- 다시 만든 뒤에는 `tests/test_silicon.py`를 돌리고, `kosmos_build.json`의 k를 확인하고 커밋한다.
