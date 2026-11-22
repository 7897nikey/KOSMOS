# experiments/ — 실험과 기록

CPU만으로 돌리는 분석 실험을 모은다. 데이터는 `research/design_v2_20261112/dataset/kosmos_long.csv.gz` 한 장, 코드는 `silicon.py`(pandas·numpy)다. API 키·GPU·KGSS 원자료가 필요 없다.

## 실험 폴더 구성

```
experiments/YYYYMMDD-<이름>/
  meta.json   담당·날짜·질문·상태(진행/완료/중단)·결론 한 줄  ← LOG.md가 여기서 만들어진다
  README.md   질문·가설·판정 기준(돌리기 전) → 방법 → 결과·결론(돌린 뒤)
  run.py      실행 코드 (silicon.py 사용)
  out/        run.py가 만든 표(csv)와 run.json(커밋·데이터 해시·파라미터·test 사용 여부)
```

`LOG.md`는 자동 생성 파일이다. 손으로 고치지 않는다.

## 처음 한 번: fork

중앙 레포는 **https://github.com/Great-Grace/KOSMOS** 다. 팀원은 각자 fork해서 작업하고 중앙 레포로 PR을 보낸다.

```bash
gh repo fork Great-Grace/KOSMOS --clone && cd KOSMOS     # 내 계정에 fork + 클론 (origin=내 fork, upstream=중앙)
git remote set-head upstream --auto                       # upstream/HEAD = 중앙 레포 기본 브랜치
```

`gh`가 없으면 GitHub에서 Fork 버튼을 누르고 다음처럼 한다.

```bash
git clone https://github.com/<내아이디>/KOSMOS && cd KOSMOS
git remote add upstream https://github.com/Great-Grace/KOSMOS
git fetch upstream && git remote set-head upstream --auto
```

## 순서 (세팅 → 실행 → 결과 push → PR)

```bash
git fetch upstream && git switch -c exp/<날짜>-<이름> upstream/HEAD   # 1. 최신 중앙 기준으로 실험 브랜치
python experiments/new.py <이름> --author 이름 --question "알고 싶은 것"
# 2. README.md에 질문·가설·판정 기준을 쓴다
git add experiments/<ID> && git commit -m "exp: <ID> 세팅" && git push -u origin HEAD
# 3. run.py를 고친다 → 커밋 → 실행
git commit -am "exp: <ID> 코드" && python experiments/<ID>/run.py
# 4. README 결과·결론, meta.json의 status·conclusion을 채운다
python experiments/index.py && python experiments/index.py --check
git add experiments && git commit -m "exp: <ID> <결론 한 줄>" && git push
# 5. 중앙 레포로 PR (템플릿 체크리스트가 뜬다). CI가 같은 점검을 다시 돌린다
gh pr create --repo Great-Grace/KOSMOS --fill
```

- push는 항상 내 fork(`origin`)로 한다. 중앙 레포에는 PR로만 들어간다.
- 2단계에서 세팅 커밋을 먼저 push하는 이유: 결과를 보기 전에 기준을 정했다는 기록이 원격에 남는다.
- 3단계에서 코드를 먼저 커밋해야 `run.json`의 `git_dirty`가 false가 된다. true인 채로는 `--check`가 `완료`를 막는다.
- PR을 올리기 전에 중앙이 바뀌었으면 `git fetch upstream && git merge upstream/HEAD`를 한다. `LOG.md`가 충돌하면 손으로 고치지 말고 `python experiments/index.py`를 다시 돌려 커밋한다.
- 처음 PR을 보내는 사람의 CI는 레포 주인이 승인해야 돈다(GitHub 기본 설정). 그 전에는 로컬 `--check` 결과를 PR에 적는다.

## `index.py --check`가 막는 것

- `meta.json`이 없거나, 필드가 빠졌거나, 폴더 이름과 id가 다른 경우
- `완료`인데 결론이 비었거나 `out/run.json`이 없는 경우
- 커밋 안 된 코드로 만든 결과(`git_dirty`), test 문항을 연 결과(`include_test`)
- 5MB가 넘는 파일(원자료·대용량은 올리지 않는다)
- `LOG.md`가 `meta.json`과 맞지 않는 경우

## 규칙

- **test 49문항은 쓰지 않는다.** `sl.load()`가 기본으로 뺀다. 설정 고르기는 calibration, 확인은 validation.
- 기각·실패한 실험도 폴더를 남긴다. `meta.json`의 상태만 `중단`/`완료`로 바꾼다.
- PR 하나에는 실험 하나만 넣는다. `dataset/`이나 `silicon.py`를 바꿔야 하면 실험과 따로 PR을 연다.
- 다른 실험 폴더의 결과를 고치지 않는다. 다시 하려면 새 실험을 만든다.
- 데이터 표를 손으로 고치지 않는다. 바꾸려면 `dataset/build_dataset.py`를 고치고 다시 만든다(원응답 필요).

## 통계 정책 (2026-11-23부터 새 실험에 적용)

- 구간: 같은 문항끼리 짝지은 부트스트랩 95%, 2,000회.
- 주 지표: 실험마다 1~2개를 README의 "돌리기 전" 칸에 미리 적는다. 나머지는 "탐색"으로 표기한다.
- 다중비교: 한 실험의 주 가설군 안에서 Holm 보정을 쓴다(부트스트랩 p값 = 2 × min(비율(차이 ≤ 0), 비율(차이 ≥ 0))). 탐색 지표는 보정하지 않는 대신 "탐색"으로 적는다.
- 보정 β: `20261123-beta-subgroup`의 규칙을 쓴다. 모델 설정 × 페르소나 유무(페르소나 없음 하나, 한국 성인과 성별×연령 셀 하나)이고 문체끼리 공유한다. β는 calibration 148문항에서 고른다.
- test 49문항: 마지막 확인 1회 전까지 열지 않는다.

## 실험 아이디어 (아직 아무도 안 함)

- 문항 유형(`scale_type`)·주제(`topic`)·민감도(`sensitivity_level`)별로 β를 따로 두면 validation TVD가 줄어드는가
- 보정을 β 하나 대신 (β, 균등 혼합 λ)로 하면 나아지는가
- ~~셀별 β(L1 10셀 각각) vs 공통 β~~ → `20261123-beta-subgroup`에서 함
- 세 모델 분포 평균(앙상블)이 단일 모델보다 사람에 가까운가
