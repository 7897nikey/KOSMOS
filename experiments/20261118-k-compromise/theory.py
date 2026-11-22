"""k 타협안의 수학적 점검 (사후 분석, API 호출 없음).
1) 사람 쪽 표본 크기(n_eff): 모델 k와 사람 n의 잡음 비교  sd(p̂_모델 − p̂_사람) ∝ sqrt(1/k + 1/n)
2) 셀 쌍 부호 일치 이론값: k개 부분표본 평균의 k=100 대비 분산 = s²(1/k − 1/100) (유한모집단 보정)
   뒤집힐 확률 = Φ(−|d| / sqrt((s_i² + s_j²)(1/k − 1/100))) → 모의실험 값과 대조
3) 모델 자체 데이터로 경험적 베이즈 τ: 셀 간 실제 분산 = 관측 분산 − p(1−p)/k,  Dirichlet 정밀도 τ = p(1−p)/실제분산 − 1
"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl
df = sl.load()
out = {}
# 1) 사람 n_eff
h = df[df.model == "deepseek"].groupby(["level", "persona", "var"]).human_neff.first().reset_index()
out["human_neff"] = {lv: {"median": float(g.human_neff.median()), "p10": float(g.human_neff.quantile(.1)), "min": float(g.human_neff.min())}
                     for lv, g in h.groupby("level")}
noise = {}
for lv in ["L0", "L1"]:
    n = out["human_neff"][lv]["median"]
    noise[lv] = {k: round(math.sqrt(1/k + 1/n) / math.sqrt(1/100 + 1/n), 3) for k in [10, 20, 30, 50, 100]}
out["noise_ratio_vs_k100"] = noise
# 2) 부호 일치 이론값
from statistics import NormalDist
Phi = NormalDist().cdf
d = df[(df.level == "L1") & (df.scale_type == "ordinal_monotone")].copy()
d["w"] = d["count"] / d.groupby(["model", "persona", "var"])["count"].transform("sum")
g = d.groupby(["model", "var", "persona"])
mean = (d.opt_idx * d.w).groupby([d.model, d["var"], d.persona]).sum()
m2 = (d.opt_idx**2 * d.w).groupby([d.model, d["var"], d.persona]).sum()
var = (m2 - mean**2).clip(lower=0)
M, V = mean.unstack("persona"), var.unstack("persona")
cols = list(M.columns)
sim = pd.read_csv(HERE / "out/summary.csv")
sim = sim[(sim.method == "raw") & (sim.level == "L1")].set_index(["model", "k"]).sign_agree
pred = {}
for model in ["exaone", "deepseek", "haiku"]:
    mm, vv = M.loc[model], V.loc[model]
    for k in [5, 10, 15, 20, 30, 40, 50]:
        f = 1/k - 1/100; ps = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                dd = (mm[cols[i]] - mm[cols[j]]).abs(); s = np.sqrt((vv[cols[i]] + vv[cols[j]]) * f)
                keep = dd >= 0.1
                z = (dd[keep] / s[keep].replace(0, np.nan)).fillna(np.inf)
                ps += [1 - Phi(-x) if np.isfinite(x) else 1.0 for x in z]
        pred[f"{model}|{k}"] = {"theory": round(float(np.mean(ps)), 3), "simulated": round(float(sim.loc[(model, k)]), 3)}
out["sign_agree_theory_vs_sim"] = pred
# 3) 경험적 베이즈 τ (모델 자체 L1 개수, 사람 정보 안 씀)
l1 = df[df.level == "L1"].copy()
l1["k"] = l1.groupby(["model", "persona", "var"])["count"].transform("sum")
l1["p"] = l1["count"] / l1["k"].clip(lower=1)
taus = {}
for model, gm in l1.groupby("model"):
    num = den = 0.0
    for (v, o), go in gm.groupby(["var", "opt_idx"]):
        pbar = go.p.mean(); obs = go.p.var(ddof=1); samp = (pbar*(1-pbar)/go.k.clip(lower=1)).mean()
        num += pbar*(1-pbar); den += max(obs - samp, 0)
    taus[model] = {"tau_EB": round(num/den - 1, 2) if den > 0 else None,
                   "weight_on_cell_at_k20": round(20/(20 + num/den - 1), 3) if den > 0 else None}
out["tau_empirical_bayes"] = taus
json.dump(out, open(HERE / "out/theory.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
