# experiments/ — 실험과 기록

CPU만으로 돌리는 분석 실험을 모은다. 데이터는 `research/design_v2_20261112/dataset/kosmos_long.csv.gz` 한 장, 코드는 `silicon.py`(pandas·numpy)다. API 키·GPU·KGSS 원자료가 필요 없다.

## 순서

1. `python experiments/new.py <이름> --author 이름 --question "알고 싶은 것"` → `experiments/YYYYMMDD-<이름>/`와 `LOG.md` 한 줄이 생긴다.
2. **돌리기 전에** `README.md`의 질문·가설·판정 기준을 쓰고 커밋한다(사후에 기준을 바꾸지 않았다는 기록).
3. `run.py`를 고쳐 실행한다. `sl.save_run`이 `out/`에 표(csv)와 `run.json`(커밋 해시, 작업트리 변경 여부, 데이터 해시, 파라미터, 시각)을 남긴다.
4. `README.md`의 결과·결론을 `out/` 숫자로 채우고, `LOG.md`의 결론·상태를 고친 뒤 커밋한다.

`run.json`의 `git_dirty`가 true면 커밋 안 된 코드로 돌린 것이다. 최종 결과는 코드를 커밋한 뒤 다시 돌려 남긴다.

## 규칙

- **test 49문항은 쓰지 않는다.** `sl.load()`가 기본으로 뺀다. 설정 고르기는 calibration, 확인은 validation.
- 기각·실패한 실험도 폴더와 LOG 줄을 남긴다. 상태만 `중단`/`완료`로 바꾼다.
- 다른 실험 폴더의 결과를 고치지 않는다. 다시 하려면 새 실험을 만든다.
- 데이터 표를 손으로 고치지 않는다. 바꾸려면 `dataset/build_dataset.py`를 고치고 다시 만든다(원응답 필요).

## 실험 아이디어 (아직 아무도 안 함)

- 문항 유형(`scale_type`)·주제(`topic`)·민감도(`sensitivity_level`)별로 β를 따로 두면 validation TVD가 줄어드는가
- 보정을 β 하나 대신 (β, 균등 혼합 λ)로 하면 나아지는가
- 셀별 β(L1 10셀 각각) vs 공통 β
- 세 모델 분포 평균(앙상블)이 단일 모델보다 사람에 가까운가
