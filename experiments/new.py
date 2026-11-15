"""새 실험 폴더를 만들고 LOG.md를 다시 만든다.

  python experiments/new.py <짧은-영문-이름> --author 이름 --question "알고 싶은 것 한 줄"
  → experiments/YYYYMMDD-<이름>/ (README.md, run.py, meta.json) 생성, LOG.md 갱신
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("slug")
ap.add_argument("--author", required=True)
ap.add_argument("--question", default="")
a = ap.parse_args()
if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", a.slug):
    raise SystemExit("이름은 영문 소문자·숫자·하이픈만 (예: beta-by-topic)")

today = datetime.date.today()
eid = f"{today:%Y%m%d}-{a.slug}"
d = HERE / eid
if d.exists():
    raise SystemExit(f"이미 있다: {d}")
d.mkdir()
for f in (HERE / "_template").iterdir():
    text = f.read_text().replace("{ID}", eid).replace("{AUTHOR}", a.author).replace("{DATE}", today.isoformat())
    (d / f.name).write_text(text)

meta = {"id": eid, "author": a.author, "date": today.isoformat(), "question": a.question,
        "status": "진행", "conclusion": ""}
(d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")
subprocess.run([sys.executable, str(HERE / "index.py")], check=True)
print(f"만듦: {d}")
print(f"다음: (브랜치 exp/{eid}에서) README.md 질문·가설 작성·커밋 → run.py 실행 → "
      "README 결과 + meta.json(status·conclusion) → index.py → 커밋 → fork에 push → 중앙(Great-Grace/KOSMOS)으로 PR")
