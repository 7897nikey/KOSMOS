"""새 실험 폴더를 만들고 LOG.md에 한 줄 추가한다.

  python experiments/new.py <짧은-영문-이름> --author 이름 --question "알고 싶은 것 한 줄"
  → experiments/YYYYMMDD-<이름>/ (README.md, run.py) 생성
"""
import argparse
import datetime
import re
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

log = HERE / "LOG.md"
q = a.question.replace("|", "/") or "(README에 작성)"
log.write_text(log.read_text().rstrip("\n") + f"\n| {today} | [{eid}]({eid}/) | {a.author} | {q} | | 진행 |\n")
print(f"만듦: {d}\n다음: README.md의 질문·가설을 먼저 쓰고, run.py를 고쳐 돌린 뒤 결과·결론과 LOG.md 상태를 채운다.")
