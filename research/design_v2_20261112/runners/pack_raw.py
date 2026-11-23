#!/usr/bin/env python3
"""원응답을 저장소용 gzip으로 묶는다 — 실행 (KOSMOS 루트): python3 research/design_v2_20261112/runners/pack_raw.py raw/v2/<이름>.jsonl

- results/raw_gz/v2/<이름>.jsonl.gz 를 만든다. 같은 이름의 .log도 함께 복사한다.
- GitHub 한 파일 한도(100MB) 때문에 압축본이 90MB를 넘으면 줄 단위로 나눠
  <이름>.jsonl.part01.gz, part02.gz … 로 만든다. 푸는 법: cat <이름>.jsonl.part*.gz | gzip -dc > raw/v2/<이름>.jsonl
- 원본 줄 수와 압축본 줄 수가 같은지 확인하고 끝낸다.
"""
import gzip
import shutil
import sys
from pathlib import Path

LIMIT = 90 * 2**20
src = Path(sys.argv[1])
dst_dir = Path(__file__).resolve().parents[3] / "results" / "raw_gz" / "v2"
dst_dir.mkdir(parents=True, exist_ok=True)
lines = src.read_bytes().splitlines(keepends=True)
one = gzip.compress(b"".join(lines), 9)
if len(one) <= LIMIT:
    outs = [dst_dir / f"{src.name}.gz"]
    outs[0].write_bytes(one)
else:
    n_parts = len(one) // LIMIT + 1
    step = len(lines) // n_parts + 1
    outs = []
    for i in range(n_parts):
        o = dst_dir / f"{src.name}.part{i + 1:02d}.gz"
        o.write_bytes(gzip.compress(b"".join(lines[i * step:(i + 1) * step]), 9))
        outs.append(o)
back = sum(len(gzip.decompress(o.read_bytes()).splitlines()) for o in outs)
assert back == len(lines), (back, len(lines))
log = src.with_suffix(".log")
if log.exists():
    shutil.copy(log, dst_dir / log.name)
for o in outs:
    print(f"{o.relative_to(dst_dir.parents[2])}  {o.stat().st_size / 2**20:.1f}MB")
print(f"줄 수 {len(lines)} 확인")
