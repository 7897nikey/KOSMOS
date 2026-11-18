#!/usr/bin/env python3
"""모델 원응답 jsonl → 프롬프트별 보기 개수 벡터 CSV.

api_error 줄은 버리고, (probe_id, sample_idx)가 겹치면 먼저 나온 것을 쓴다.
DeepSeek(run_deepseek_eco.py)와 K-EXAONE 병합본(merge_check.py 출력) 모두 읽는다.
인칭·화계 결과(--prompts register, probe_id 끝이 __B/__C/__D)도 읽는다. 이때 확정본
register_v1/out/prompts_register_v1.csv의 해시를 summary.json과 대조한다. cond 열은 v2 프롬프트면 A다.

사용 (KOSMOS 루트에서):
  python3 research/design_v2_20261112/runners/aggregate_counts.py raw/v2/deepseek_ultrafast_p1.jsonl
출력: 같은 위치에 <이름>_counts.csv
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

DESIGN = Path(__file__).resolve().parent.parent
REGISTER = DESIGN / "register_v1" / "out"


def load_probes() -> dict:
    probes = {p["probe_id"]: dict(p, cond="A") for p in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig"))}
    f = REGISTER / "prompts_register_v1.csv"
    if f.exists():
        want = json.loads((REGISTER / "summary.json").read_text(encoding="utf-8"))["prompts_sha256"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != want:
            sys.exit(f"{f.name} 해시가 summary.json과 다릅니다. 중단합니다.")
        for p in csv.DictReader(f.open(encoding="utf-8-sig")):
            if p["reuse_existing"] == "False":  # A 조건은 prompts_v2.csv의 probe_id로 이미 있다
                probes[p["probe_id"]] = p
    return probes


def main() -> None:
    src = Path(sys.argv[1])
    probes = load_probes()
    seen, recs = set(), []
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error":
            continue
        k = (r["probe_id"], r["sample_idx"])
        if k in seen:
            continue
        seen.add(k)
        recs.append(r)

    by = collections.defaultdict(list)
    for r in recs:
        by[r["probe_id"]].append(r)

    out = src.with_name(src.stem + "_counts.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_id", "phase", "level", "cell_id", "var", "codes", "n_total", "n_valid",
                    "n_refusal", "n_other_invalid", "counts_json", "cond"])
        for pid, rs in sorted(by.items()):
            if pid not in probes:
                sys.exit(f"프롬프트 목록에 없는 probe_id: {pid} (register 결과면 확정본 CSV가 있어야 한다)")
            p = probes[pid]
            codes = json.loads(p["codes"])
            c = collections.Counter(r["parsed"] for r in rs if r["valid"])
            n_ref = sum(r["invalid_type"] == "refusal" for r in rs)
            n_valid = sum(c.values())
            w.writerow([pid, p["phase"], p["level"], p["cell_id"], p["var"], p["codes"], len(rs), n_valid,
                        n_ref, len(rs) - n_valid - n_ref, json.dumps([c.get(x, 0) for x in codes]), p["cond"]])
    print(f"{out}: 프롬프트 {len(by)}개, 응답 {len(recs)}건")


if __name__ == "__main__":
    main()
