"""DeepSeek Ultrafast·Haiku 5.5를 추론 low로 1단계(c0/L0/L1, k=100) 다시 돌릴 때 비용 추정.
입력·답 토큰은 2026-11-12 추론 끔 실행의 실측치, 추론 토큰은 시나리오(실측 전). 유료 호출 0건."""
import csv, json, hashlib, subprocess, sys, datetime, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]; OUT = pathlib.Path(__file__).parent
SRC = {"deepseek": ROOT/"raw/v2/deepseek_ultrafast_p1.jsonl", "haiku": ROOT/"raw/v2/haiku_p1.jsonl"}
PRICE = {  # USD / 1M tokens (in, out). 추론 토큰은 출력으로 과금
    "deepseek": {"ultrafast_-70%": (0.09, 0.36), "ultrafast_-30%": (0.21, 0.84), "ultrafast_정가": (0.30, 1.20)},
    "haiku": {"batch": (0.05, 0.25), "standard": (0.10, 0.50)},
}
REASONING = [0, 50, 170, 215, 500, 1000]  # 호출당 추론 토큰. 170/215 = K-EXAONE low 실측 중앙값/평균
meas = {}
for m, p in SRC.items():
    n = i = o = 0
    for l in open(p):
        r = json.loads(l)
        if r.get("invalid_type") == "api_error" or r.get("input_tokens") is None: continue
        n += 1; i += r["input_tokens"]; o += r["output_tokens"]
    meas[m] = {"calls": n, "in_avg": i/n, "ans_avg": o/n}
rows = []
for m, tiers in PRICE.items():
    c = meas[m]
    for tier, (pi, po) in tiers.items():
        for R in REASONING:
            cin = c["calls"]*c["in_avg"]*pi/1e6; cout = c["calls"]*(c["ans_avg"]+R)*po/1e6
            rows.append({"model": m, "tier": tier, "reasoning_per_call": R, "calls": c["calls"],
                         "input_usd": round(cin, 2), "output_usd": round(cout, 2), "total_usd": round(cin+cout, 2)})
with open(OUT/"estimate.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
commit = subprocess.run(["git","-C",str(ROOT),"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": commit,
           "inputs": {m: {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for m, p in SRC.items()},
           "measured": meas, "price": PRICE, "reasoning_scenarios": REASONING, "paid_calls": 0},
          open(OUT/"run.json", "w"), ensure_ascii=False, indent=1)
for r in rows: print(r)
