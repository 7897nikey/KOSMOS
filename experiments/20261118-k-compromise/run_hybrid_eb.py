"""사후 확인 2: C안(c0·L0 k=50 + L1 k=20 + 축소)을 τ를 사람 정보 없이 모델 자체 데이터의 경험적 베이즈 값으로 고정해 다시 잰다.
τ_EB는 out/theory.json (DeepSeek 4.64, EXAONE 7.12, Haiku 0.88). 비교용으로 축소 없음(τ=0)도 같은 표본에서 잰다."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
src = (HERE / "run.py").read_text(encoding="utf-8").split("# 기준선: 전체 데이터")[0]
g = {"__file__": str(HERE / "run.py")}; exec(compile(src, "run.py(head)", "exec"), g)
sl, df, grp, counts = g["sl"], g["df"], g["grp"], g["counts"]
TAU = {m: v["tau_EB"] for m, v in json.load(open(HERE / "out/theory.json"))["tau_empirical_bayes"].items()}
lv_of = {k: df.loc[idx[0], "level"] for k, idx in grp.items()}
full = g["evaluate"](counts, g["fit"](counts))
def shrink_eb(c):
    d = g["with_counts"](c); l1 = d["level"] == "L1"
    pooled = d[l1].groupby(["model", "var", "opt_idx"])["count"].transform("sum")
    tot = d[l1].groupby(["model", "var"])["count"].transform("sum")
    tau = d.loc[l1, "model"].map(TAU)
    out = c.copy(); out[l1.to_numpy()] = c[l1.to_numpy()] + (tau * pooled / tot.clip(lower=1)).to_numpy()
    return out
rows = []
for rep in range(10):
    rng = np.random.default_rng(30_000 + rep)   # run_hybrid.py와 같은 표본
    new = counts.copy()
    for key, idx in grp.items():
        k = 20 if lv_of[key] == "L1" else 50
        c = counts[idx].astype(int)
        if c.sum() > k: new[idx] = rng.multivariate_hypergeometric(c, k)
    for name, cc in [("no_shrink", new), ("shrink_EB", shrink_eb(new))]:
        b = g["fit"](cc); s = g["evaluate"](cc, b)
        for m in g["MODELS"]:
            for lv in ["L0", "L1"]:
                rows.append(dict(est=name, rep=rep, model=m, level=lv, d_tvd=s.loc[m, lv] - full.loc[m, lv]))
r = pd.DataFrame(rows)
t = r.groupby(["est", "model", "level"]).d_tvd.agg(["mean", "max"]).round(4)
print(t); print("τ_EB", TAU)
sl.save_run(HERE / "out_hybrid_eb", params={"tau_eb": TAU, "k_top": 50, "k_l1": 20, "post_hoc": True},
            tables={"summary": t, "reps": r.set_index(["est", "rep", "model", "level"])})
