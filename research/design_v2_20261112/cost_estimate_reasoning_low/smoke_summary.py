"""추론 low 스모크 결과 집계 → 1단계 전체(295,200건) 비용 외삽. 사용: python3 smoke_summary.py"""
import json, statistics as st, collections, pathlib, datetime, hashlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; OUT = pathlib.Path(__file__).parent
N_FULL = 295200
RUNS = {"deepseek_eco_low": (ROOT/"raw/v2/deepseek_eco_low_p1_smoke.jsonl", (0.03, 0.12)),
        "haiku_low_batch": (ROOT/"raw/v2/haiku_low_p1_smoke.jsonl", (0.05, 0.25)),
        "haiku_medium_batch": (ROOT/"raw/v2/haiku_medium_p1_smoke.jsonl", (0.05, 0.25))}
res = {}
for name, (p, (pi, po)) in RUNS.items():
    if not p.exists(): res[name] = "없음"; continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    rows = [r for r in rows if r.get("invalid_type") != "api_error"]
    rt = [r.get("reasoning_tokens", r["output_tokens"]) or 0 for r in rows]
    out = [r["output_tokens"] for r in rows]; inp = [r["input_tokens"] for r in rows]
    q = st.quantiles(out, n=20)
    lv = collections.defaultdict(list)
    for r in rows: lv[r["level"]].append(r["output_tokens"])
    ain, aout = st.mean(inp), st.mean(out)
    # 단계 가중: 1단계 구성 c0 1/12, L0 1/12, L1 10/12
    w = {"c0": 1/12, "L0": 1/12, "L1": 10/12}
    aout_w = sum(w[k]*st.mean(v) for k, v in lv.items())
    res[name] = {"n": len(rows), "valid": sum(r["valid"] for r in rows)/len(rows),
                 "invalid": dict(collections.Counter(r["invalid_type"] for r in rows if not r["valid"])),
                 "in_avg": ain, "out_avg": aout, "out_avg_level_weighted": aout_w,
                 "reasoning_avg": st.mean(rt), "out_median": st.median(out), "out_p95": q[18], "out_max": max(out),
                 "out_by_level": {k: round(st.mean(v), 1) for k, v in lv.items()},
                 "smoke_cost_usd": round((sum(inp)*pi + sum(out)*po)/1e6, 4),
                 "full_cost_usd": round(N_FULL*(ain*pi + aout_w*po)/1e6, 2),
                 "full_cost_input_usd": round(N_FULL*ain*pi/1e6, 2), "full_cost_output_usd": round(N_FULL*aout_w*po/1e6, 2),
                 "price_in_out": [pi, po], "input_sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
res["_time"] = datetime.datetime.now().isoformat(timespec="seconds")
json.dump(res, open(OUT/"smoke_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
