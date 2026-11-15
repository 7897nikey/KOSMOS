"""EXAONE 중간 결과(m3 zip + m1 진행 중 파일)로 표본 번호를 맞춘 3모델 개수 CSV를 만든다."""
import collections, csv, json, sys, zipfile, io
from pathlib import Path
ROOT = Path("/Users/taewoo/Developer/KOSMOS"); D = ROOT/"research/design_v2_20261112"; OUT = Path(sys.argv[1])
PK = D/"team_packages/KOSMOS_EXAONE_m1_태우"
probes = {p["probe_id"]: p for p in csv.DictReader((D/"prompts_v2.csv").open(encoding="utf-8-sig")) if p["phase"] == "1"}

def read(lines, recs, seen):
    for l in lines:
        if not l.strip(): continue
        try: r = json.loads(l)
        except json.JSONDecodeError: continue  # 쓰는 중인 마지막 줄
        if r.get("invalid_type") == "api_error" or r["probe_id"] not in probes: continue
        k = (r["probe_id"], r["sample_idx"])
        if k in seen: continue
        seen.add(k); recs[k] = r

ex, seen = {}, set()
with zipfile.ZipFile(PK/"KOSMOS_EXAONE_결과_m3_허다교.zip") as z:
    read(io.TextIOWrapper(z.open("exaone_m3_p1.jsonl"), encoding="utf-8"), ex, seen)
read(open(PK/"exaone_m1_p1.jsonl", encoding="utf-8"), ex, seen)
# 모든 프롬프트에 다 있는 표본 번호만 쓴다
have = collections.defaultdict(set)
for (pid, s) in ex: have[s].add(pid)
S = sorted(s for s, ps in have.items() if len(ps) == len(probes))
print("EXAONE 표본 번호(전 프롬프트 완비):", S[:3], "...", S[-3:], "개수", len(S))
Sset = set(S)

def load(path):
    recs = {}
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("invalid_type") == "api_error" or r["probe_id"] not in probes: continue
        recs.setdefault((r["probe_id"], r["sample_idx"]), r)
    return recs

def write(name, recs):
    by = collections.defaultdict(list)
    for (pid, s), r in recs.items():
        if s in Sset: by[pid].append(r)
    with (OUT/f"{name}_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_id","phase","level","cell_id","var","codes","n_total","n_valid","n_refusal","n_other_invalid","counts_json"])
        for pid, rs in sorted(by.items()):
            p = probes[pid]; codes = json.loads(p["codes"])
            c = collections.Counter(r["parsed"] for r in rs if r["valid"])
            nr = sum(r["invalid_type"] == "refusal" for r in rs); nv = sum(c.values())
            w.writerow([pid, p["phase"], p["level"], p["cell_id"], p["var"], p["codes"], len(rs), nv, nr, len(rs)-nv-nr, json.dumps([c.get(x,0) for x in codes])])
    print(name, "프롬프트", len(by), "응답", sum(map(len, by.values())))

write("exaone", ex)
write("deepseek", load(ROOT/"raw/v2/deepseek_ultrafast_p1.jsonl"))
write("haiku", load(ROOT/"raw/v2/haiku_p1.jsonl"))
json.dump(S, open(OUT/"samples_used.json","w"))
