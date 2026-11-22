"""타협안별 DeepSeek Eco 비용 (스모크 실측 토큰, 단계별 평균). 추론 끔은 Ultrafast 1단계 실측 토큰에 Eco 단가."""
import json, collections, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
PI, PO = 0.03, 0.12
def per_level(f):
    s = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(ROOT / f):
        r = json.loads(l)
        if r.get("invalid_type") == "api_error" or r.get("input_tokens") is None: continue
        a = s[r["level"]]; a[0] += 1; a[1] += r["input_tokens"]; a[2] += r["output_tokens"]
    return {lv: (a[1]/a[0]*PI + a[2]/a[0]*PO)/1e6 for lv, a in s.items()}
low = per_level("raw/v2/deepseek_eco_low_p1_smoke.jsonl"); off = per_level("raw/v2/deepseek_ultrafast_p1.jsonl")
N = {"c0": 246, "L0": 246, "L1": 2460}
PLANS = {"k=100 전체 (현행)": {"c0": 100, "L0": 100, "L1": 100},
         "A. k=50 전체 (사전 기준 통과)": {"c0": 50, "L0": 50, "L1": 50},
         "B. c0·L0 k=50 + L1 k=40": {"c0": 50, "L0": 50, "L1": 40},
         "C. c0·L0 k=50 + L1 k=20 + 축소 (사후)": {"c0": 50, "L0": 50, "L1": 20},
         "D. c0·L0 k=30 + L1 k=20 + 축소": {"c0": 30, "L0": 30, "L1": 20}}
out = []
for name, ks in PLANS.items():
    calls = sum(N[l]*ks[l] for l in N)
    cl = sum(N[l]*ks[l]*low[l] for l in N); co = sum(N[l]*ks[l]*off[l] for l in N)
    out.append(dict(plan=name, calls=calls, share=round(calls/295200, 3), eco_low=round(cl, 2), eco_off=round(co, 2), paired=round(cl+co, 2)))
    print(out[-1])
json.dump(out, open(pathlib.Path(__file__).parent/"out/cost_plans.json", "w"), ensure_ascii=False, indent=1)
