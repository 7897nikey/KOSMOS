"""20261119-register-effect (E1) — DeepSeek Eco 인칭×화계, 추론 끔·low, k=50.
실행 (KOSMOS 루트): python experiments/20261119-register-effect/run.py

1. 원응답 → 개수 C[설정][조건][단계(c0/L0), 문항, 보기] (표본 0~49, 중복·api_error 제외)
2. Q1~Q5 문항별 값
3. 같은 문항끼리 짝지은 부트스트랩 95% 구간
"""
import csv
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS_V2 = DESIGN / "prompts_v2.csv"
PROMPTS_REG = DESIGN / "register_v1/out/prompts_register_v1.csv"
REG_SHA = "1d1530ca17ee84b15fc1728ede4317541fc90d5ba212223e849a0073f7d06636"
RAW = {"끔": {"A": ROOT / "raw/v2/deepseek_eco_p1.jsonl", "BCD": ROOT / "raw/v2/deepseek_eco_register_p1.jsonl"},
       "low": {"A": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl", "BCD": ROOT / "raw/v2/deepseek_eco_low_register_p1.jsonl"},
       # 2026-11-19 추가: Haiku 5.5 추론 끔. 원문은 1단계(k=100)의 표본 0~49, 새 문체는 Batch API로 같은 설정(max_tokens 5).
       "haiku": {"A": ROOT / "raw/v2/haiku_p1.jsonl", "BCD": ROOT / "raw/v2/haiku_register_p1.jsonl"}}
K, B, NULL_REPS = 50, 2000, 200
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
rng = np.random.default_rng(20261119)

out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# ── 1. 데이터 ───────────────────────────────────────────
if hashlib.sha256(PROMPTS_REG.read_bytes()).hexdigest() != REG_SHA:
    sys.exit("register 프롬프트 해시가 확정본과 다름")
d = np.load(NPZ, allow_pickle=True)
H_ALL, CODES, SPLIT = d["H"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
H = H_ALL[:2]                                  # c0, L0 (둘 다 모집단 전체)
MASK = ~np.isnan(CODES)
EV, CAL, VAL = SPLIT != "test", SPLIT == "calibration", SPLIT == "validation"
LV = {"c0": 0, "L0": 1}

meta = {}
for r in csv.DictReader(PROMPTS_V2.open(encoding="utf-8-sig")):
    if r["phase"] == "1" and r["level"] in LV:
        meta[r["probe_id"]] = (r["level"], r["var"], "A")
for r in csv.DictReader(PROMPTS_REG.open(encoding="utf-8-sig")):
    if r["reuse_existing"] == "False":
        meta[r["probe_id"]] = (r["level"], r["var"], r["cond"])


def load(paths):
    C = {c: np.zeros((2, len(VARS), CODES.shape[1])) for c in "ABCD"}
    n = {c: np.zeros((2, len(VARS))) for c in "ABCD"}
    ref = {c: 0 for c in "ABCD"}
    seen = set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            key = (r["probe_id"], r["sample_idx"])
            if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in meta:
                continue
            seen.add(key)
            lv, var, c = meta[r["probe_id"]]
            j, i = LV[lv], VARS.index(var)
            n[c][j, i] += 1
            if r["valid"]:
                C[c][j, i, list(CODES[i]).index(float(r["parsed"]))] += 1
            elif r.get("invalid_type") == "refusal":
                ref[c] += 1
    return C, n, ref


def prob(X, beta=1.0):
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


top = lambda x: np.where(MASK, x, -np.inf).argmax(-1)


def noise_tvd(Ca, Cb):
    """두 조건 개수를 합친 분포에서 각자 표본 수만큼 다시 뽑았을 때의 TVD 평균(문항별)."""
    pooled = share(Ca + Cb)
    na, nb = Ca.sum(-1).astype(int), Cb.sum(-1).astype(int)
    res = np.zeros(len(VARS))
    for i in range(len(VARS)):
        k_ = MASK[i].sum()
        p = pooled[i, :k_]
        if na[i] == 0 or nb[i] == 0:
            res[i] = np.nan
            continue
        a = rng.multinomial(na[i], p, size=NULL_REPS) / na[i]
        b = rng.multinomial(nb[i], p, size=NULL_REPS) / nb[i]
        res[i] = tvd(a, b).mean()
    return res


def calib_tvd(Cl):
    """Cl [문항, 보기]: β를 calibration에서 골라 문항별 TVD."""
    g = tvd(prob(Cl[None], BETAS[:, None, None]), H[1][None])   # [β, 문항]
    return g[g[:, CAL].mean(1).argmin()], float(BETAS[g[:, CAL].mean(1).argmin()])


def boot(x, sel):
    idx = np.where(sel & ~np.isnan(x))[0]
    bs = x[idx[rng.integers(0, len(idx), (B, len(idx)))]].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(x[idx].mean()), float(lo), float(hi), len(idx)


rows, info = [], {}


def add(setting, q, label, x, sel, ref_val=None):
    m, lo, hi = boot(x, sel)[:3]
    rows.append(dict(설정=setting, 질문=q, 비교=label, 값=m, 하한=lo, 상한=hi, 유의=bool(lo > 0 or hi < 0),
                     **({"참고": ref_val} if ref_val is not None else {})))


for setting, paths in RAW.items():
    C, n, ref = load([paths["A"], paths["BCD"]])
    for c in "ABCD":
        nn = n[c][0 if c == "A" or c == "B" else 1]
        tot = n[c][[0, 1] if c in "AB" else [1]].sum()
        val = C[c][[0, 1] if c in "AB" else [1]].sum()
        info[f"{setting} {c}"] = dict(응답=int(tot), 유효율=round(val / tot, 4), 거부=ref[c],
                                     표본최소=int(n[c][[0, 1] if c in "AB" else [1]].min()))
    log(f"[{setting}]", {c: info[f'{setting} {c}'] for c in "ABCD"})

    # Q1 민감도: 모델끼리 TVD − 표집 잡음
    for lvn, j, conds in [("c0", 0, "B"), ("L0", 1, "BCD")]:
        for c in conds:
            obs = tvd(share(C[c][j]), share(C["A"][j]))
            nul = noise_tvd(C[c][j], C["A"][j])
            add(setting, "Q1 민감도 (모델 TVD − 잡음)", f"{lvn} {c} vs A", obs - nul, EV, ref_val=float(np.nanmean(obs[EV])))
            sig_items = float(np.mean((obs - nul)[EV] > 0))
            info[f"{setting} Q1 {lvn} {c}"] = dict(관측평균=float(np.nanmean(obs[EV])), 잡음평균=float(np.nanmean(nul[EV])),
                                                   잡음보다큰문항비율=round(sig_items, 3))

    # 사람 TVD (β=1)
    T = {(lvn, c): tvd(prob(C[c][j]), H[j]) for lvn, j in LV.items() for c in "ABCD" if not (lvn == "c0" and c in "CD")}
    for (lvn, c), t in T.items():
        rows.append(dict(설정=setting, 질문="수준 사람 TVD(β=1)", 비교=f"{lvn} {c}", 값=float(t[EV].mean()), 하한=np.nan, 상한=np.nan, 유의=False))
    # Q2 재현
    for (lvn, c), t in T.items():
        if c != "A":
            add(setting, "Q2 사람 TVD(β=1) X−A", f"{lvn} {c}−A", t - T[(lvn, "A")], EV)
    # Q2' 보정 재현 (L0)
    TC = {}
    for c in "ABCD":
        TC[c], beta = calib_tvd(C[c][1])
        rows.append(dict(설정=setting, 질문="수준 보정 TVD(L0, validation)", 비교=f"L0 {c} (β={beta})", 값=float(TC[c][VAL].mean()),
                         하한=np.nan, 상한=np.nan, 유의=False))
    for c in "BCD":
        add(setting, "Q2' 보정 TVD(L0, validation) X−A", f"L0 {c}−A", TC[c] - TC["A"], VAL)
    # Q3 L0 2×2
    a_, b_, c_, d_ = (T[("L0", x)] for x in "ABCD")
    add(setting, "Q3 L0 2×2 (사람 TVD)", "인칭 효과 (이 사람 − 당신)", (c_ + d_ - a_ - b_) / 2, EV)
    add(setting, "Q3 L0 2×2 (사람 TVD)", "화계 효과 (해라 − 하십시오)", (b_ + d_ - a_ - c_) / 2, EV)
    add(setting, "Q3 L0 2×2 (사람 TVD)", "상호작용 (D−C)−(B−A)", (d_ - c_) - (b_ - a_), EV)
    # Q4 최빈값 일치 L0
    M = {c: (top(share(C[c][1])) == top(H[1])).astype(float) for c in "ABCD"}
    for c in "ABCD":
        rows.append(dict(설정=setting, 질문="수준 최빈값 일치(L0)", 비교=f"L0 {c}", 값=float(M[c][EV].mean()), 하한=np.nan, 상한=np.nan, 유의=False))
    for c in "BCD":
        add(setting, "Q4 최빈값 일치(L0) X−A", f"L0 {c}−A", M[c] - M["A"], EV)
    # Q5 페르소나 효과 c0→L0
    pa, pb = T[("L0", "A")] - T[("c0", "A")], T[("L0", "B")] - T[("c0", "B")]
    rows.append(dict(설정=setting, 질문="수준 c0→L0 변화", 비교="A 하십시오", 값=float(pa[EV].mean()), 하한=np.nan, 상한=np.nan, 유의=False))
    rows.append(dict(설정=setting, 질문="수준 c0→L0 변화", 비교="B 해라", 값=float(pb[EV].mean()), 하한=np.nan, 상한=np.nan, 유의=False))
    add(setting, "Q5 c0→L0 변화 B−A", "해라 − 하십시오", pb - pa, EV)

res = pd.DataFrame(rows)
res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
log(res.round(4).to_string(index=False))
log(json.dumps({k: v for k, v in info.items() if "Q1" in k}, ensure_ascii=False, indent=1))
logf.close()  # git 상태를 보기 전에 run.log를 다 써 둔다(덜 쓰인 파일이 dirty로 잡히지 않게)

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "inputs": {f"{s} {k}": {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                      for s, ps in RAW.items() for k, p in ps.items()} | {"prompts_register": REG_SHA[:16]},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "beta_grid": [0.05, 1.5, 0.005]},
           "include_test": False, "info": info}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
