## 실험

- 실험 ID: `experiments/YYYYMMDD-이름/`
- 질문:
- 결론 (meta.json의 conclusion):

## 점검

- [ ] 질문·가설·판정 기준을 결과 보기 **전에** 커밋했다
- [ ] 코드를 커밋한 뒤 `run.py`를 다시 돌렸다 (`out/run.json`의 `git_dirty: false`)
- [ ] test 문항을 쓰지 않았다 (`include_test: false`)
- [ ] `python experiments/index.py --check` 통과
- [ ] 다른 실험 폴더, `dataset/`, 설계 고정 파일은 건드리지 않았다
