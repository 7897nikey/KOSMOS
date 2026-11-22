"""스모크 실측 토큰 기준 1단계(2,952프롬프트) 비용을 K별로 정리. 유료 호출 없음.
- 단가(USD/1M, 입력/출력): DeepSeek Eco $0.03/$0.12(상시 90% 할인가), Haiku Batch $0.05/$0.25(50% 할인).
- 캐시: DeepSeek Eco 스모크 cached_tokens 0/48,537 → 미적용. Haiku 5.5 최소 캐시 길이 512토큰 > 프롬프트 평균 207 → 미적용.
  가정 열: 프롬프트를 512토큰으로 늘려 캐시를 걸고 배치 안 적중률 100%(쓰기 1회 1.25배, 이후 K-1회 0.1배). 프롬프트가 바뀌므로 참고용.
- 추론 끔 행: DeepSeek은 Ultrafast 1단계 실측 토큰에 Eco 단가, Haiku는 1단계 실측.
- 구간: 호출당 출력 토큰 평균의 부트스트랩 95% CI(2,000회).
"""
import json, csv, pathlib, random, datetime
ROOT = pathlib.Path(__file__).resolve().parents[3]; OUT = pathlib.Path(__file__).parent
P = 2952; KS = [5, 10, 20, 30, 50, 100]
def load(f):
    rows = [json.loads(l) for l in open(ROOT/f) if l.strip()]
    return [r for r in rows if r.get("invalid_type") != "api_error" and r.get("input_tokens") is not None]
def mean(x): return sum(x)/len(x)
rng = random.Random(20261118)
def ci(x):
    b = sorted(mean([x[rng.randrange(len(x))] for _ in x]) for _ in range(2000)); return b[49], b[1949]
ARMS = [
    ("DeepSeek Eco 추론 low", "raw/v2/deepseek_eco_low_p1_smoke.jsonl", 0.03, 0.12, False),
    ("Haiku Batch 추론 low", "raw/v2/haiku_low_p1_smoke.jsonl", 0.05, 0.25, True),
    ("Haiku Batch 추론 medium", "raw/v2/haiku_medium_p1_smoke.jsonl", 0.05, 0.25, True),
    ("(참고) DeepSeek Eco 추론 끔", "raw/v2/deepseek_ultrafast_p1.jsonl", 0.03, 0.12, False),
    ("(참고) Haiku Batch 추론 끔", "raw/v2/haiku_p1.jsonl", 0.05, 0.25, True),
]
rows = []
for name, f, pi, po, haiku in ARMS:
    r = load(f); ain = mean([x["input_tokens"] for x in r]); outs = [x["output_tokens"] for x in r]
    aout = mean(outs); lo, hi = ci(outs) if len(outs) <= 5000 else (aout, aout)
    for K in KS:
        n = P*K
        cin = n*ain*pi/1e6
        row = {"arm": name, "K": K, "calls": n, "in_avg": round(ain, 1), "out_avg": round(aout, 2),
               "input_usd": round(cin, 3), "output_usd": round(n*aout*po/1e6, 3),
               "total_usd": round(cin + n*aout*po/1e6, 2),
               "total_lo": round(cin + n*lo*po/1e6, 2), "total_hi": round(cin + n*hi*po/1e6, 2)}
        if haiku:  # 캐시 가정: 512토큰 프롬프트, 프롬프트당 쓰기 1회(1.25배) + 읽기 K-1회(0.1배)
            cin_c = P*512*pi*(1.25 + 0.1*(K-1))/1e6
            row["total_cache_hyp_usd"] = round(cin_c + n*aout*po/1e6, 2)
        rows.append(row)
keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
with open(OUT/"cost_by_k.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "prompts": P, "ks": KS,
           "arms": [a[:4] for a in ARMS], "paid_calls": 0}, open(OUT/"cost_by_k.run.json", "w"), ensure_ascii=False, indent=1)
for r in rows: print(r)
