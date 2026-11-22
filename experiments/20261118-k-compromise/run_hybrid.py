"""사후 확인(판정 기준을 본 뒤 추가): L0·c0는 k=50 재표본, L1은 k=20 재표본 + 축소(τ calibration에서 선택).
run.py의 함수를 그대로 쓴다. 실행: python experiments/20261118-k-compromise/run_hybrid.py"""
import runpy, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
src = (HERE / "run.py").read_text(encoding="utf-8").split("# 기준선: 전체 데이터")[0]
g = {"__file__": str(HERE / "run.py")}
exec(compile(src, "run.py(head)", "exec"), g)
sl, np_, df, grp, counts = g["sl"], g["np"], g["df"], g["grp"], g["counts"]
K_TOP, K_L1, REPS = 50, 20, 10
keys = list(grp.keys())
lv_of = {k: df.loc[idx[0], "level"] for k, idx in grp.items()}
beta_full = g["fit"](counts)
full = g["evaluate"](counts, beta_full)
rows, cmp = [], []
for rep in range(REPS):
    rng = np.random.default_rng(30_000 + rep)
    new = counts.copy()
    for key, idx in grp.items():
        k = K_L1 if lv_of[key] == "L1" else K_TOP
        c = counts[idx].astype(int)
        if c.sum() > k:
            new[idx] = rng.multivariate_hypergeometric(c, k)
    best = None
    for tau in g["TAUS"]:
        cs = g["shrink"](new, tau); b = g["fit"](cs)
        cal = g["with_counts"](cs)
        v = sl.summary(cal[cal["split"] == "calibration"], b)["L1"].mean()
        if best is None or v < best[0] - 1e-12: best = (v, tau, cs, b)
    s = g["evaluate"](best[2], best[3]); agree = g["sign_agree"](best[2])
    for m in g["MODELS"]:
        for lv in sl.LEVELS:
            rows.append(dict(rep=rep, model=m, level=lv, tvd=s.loc[m, lv], d_tvd=s.loc[m, lv] - full.loc[m, lv],
                             tau=best[1], sign_agree=agree[m]))
    for a, b2 in [("exaone", "deepseek"), ("exaone", "haiku")]:
        cmp.append(dict(rep=rep, pair=f"{a}-{b2}",
                        same=bool(np.sign(s.loc[a, "L1"] - s.loc[b2, "L1"]) == np.sign(full.loc[a, "L1"] - full.loc[b2, "L1"]))))
    print("rep", rep, "tau", best[1], flush=True)
r, c = pd.DataFrame(rows), pd.DataFrame(cmp)
summ = r.groupby(["model", "level"]).agg(d_tvd=("d_tvd", "mean"), d_tvd_max=("d_tvd", "max"), sign_agree=("sign_agree", "mean")).round(4)
print(summ); print(c.groupby("pair").same.mean()); print("tau 분포", r.groupby("rep").tau.first().value_counts().to_dict())
calls = 492 * K_TOP + 2460 * K_L1
print("호출 수", calls, "(k=100 대비", round(calls / 295200, 3), ")")
sl.save_run(HERE / "out_hybrid", params={"k_top": K_TOP, "k_l1": K_L1, "reps": REPS, "calls": calls, "post_hoc": True},
            tables={"summary": summ, "model_compare": c.groupby("pair").same.mean().to_frame(), "reps": r.set_index(["rep", "model", "level"])})
