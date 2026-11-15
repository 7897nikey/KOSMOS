"""실험 목록(LOG.md)을 각 실험의 meta.json에서 다시 만들고, push 전 점검을 한다.

  python experiments/index.py           # LOG.md 다시 쓰기
  python experiments/index.py --check   # 점검만 (CI와 push 전). 문제가 있으면 종료 코드 1

LOG.md는 손으로 고치지 않는다. 여러 명이 동시에 올려 LOG.md가 충돌하면 이 스크립트를 다시 돌리면 된다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS = {"진행", "완료", "중단"}
FIELDS = ["id", "author", "date", "question", "status", "conclusion"]
MAX_MB = 5

HEAD = """# 실험 기록

자동 생성 파일이다. 손으로 고치지 말고 각 실험의 `meta.json`을 고친 뒤 `python experiments/index.py`를 돌린다.
새 실험: `python experiments/new.py <이름> --author 이름 --question "..."`. 기각된 결과도 지우지 않는다.

| 날짜 | 실험 | 담당 | 질문 | 결론 | 상태 |
|---|---|---|---|---|---|
"""


def experiments():
    return sorted(d for d in HERE.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}-[a-z0-9-]+", d.name))


def cell(s):
    return str(s).replace("|", "/").replace("\n", " ").strip()


def render(metas):
    rows = [f"| {m['date']} | [{m['id']}]({m['id']}/) | {cell(m['author'])} | {cell(m['question'])} | "
            f"{cell(m['conclusion'])} | {m['status']} |" for m in sorted(metas, key=lambda m: (m["date"], m["id"]))]
    return HEAD + "\n".join(rows) + ("\n" if rows else "")


def check(d):
    errs, meta = [], {}
    f = d / "meta.json"
    if not f.exists():
        return [f"{d.name}: meta.json 없음"], None
    try:
        meta = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        return [f"{d.name}: meta.json 형식 오류 ({e})"], None
    errs += [f"{d.name}: meta.json에 '{k}' 없음" for k in FIELDS if k not in meta]
    if errs:
        return errs, None
    if meta["id"] != d.name:
        errs.append(f"{d.name}: meta.json id가 폴더 이름과 다름 ({meta['id']})")
    if meta["status"] not in STATUS:
        errs.append(f"{d.name}: status는 {sorted(STATUS)} 중 하나 ({meta['status']})")
    for need in ["README.md", "run.py"]:
        if not (d / need).exists():
            errs.append(f"{d.name}: {need} 없음")
    if meta["status"] == "완료":
        if not meta["conclusion"].strip():
            errs.append(f"{d.name}: 완료인데 conclusion이 비어 있음")
        run = d / "out/run.json"
        if not run.exists():
            errs.append(f"{d.name}: 완료인데 out/run.json 없음 (run.py를 돌려야 함)")
        else:
            r = json.loads(run.read_text())
            if r.get("git_dirty"):
                errs.append(f"{d.name}: out/run.json이 커밋 안 된 코드로 만든 결과 (코드 커밋 후 다시 실행)")
            if r.get("include_test"):
                errs.append(f"{d.name}: test 문항을 연 결과 (팀 합의 없이 금지)")
    for p in d.rglob("*"):
        if p.is_file() and p.stat().st_size > MAX_MB * 2**20:
            errs.append(f"{d.name}: {p.relative_to(d)}가 {MAX_MB}MB 초과 (원자료·대용량은 올리지 않는다)")
    return errs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    errs, metas = [], []
    for d in experiments():
        e, m = check(d)
        errs += e
        if m:
            metas.append(m)
    text = render(metas)
    log = HERE / "LOG.md"
    if a.check:
        if not log.exists() or log.read_text() != text:
            errs.append("LOG.md가 meta.json과 다름 → python experiments/index.py 실행 후 커밋")
        for e in errs:
            print("✗", e)
        print("점검 통과" if not errs else f"\n문제 {len(errs)}개")
        sys.exit(1 if errs else 0)
    log.write_text(text)
    print(f"LOG.md 갱신: 실험 {len(metas)}개")
    for e in errs:
        print("⚠", e)


if __name__ == "__main__":
    main()
