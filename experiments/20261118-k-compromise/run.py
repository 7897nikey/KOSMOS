"""20261118-k-compromise — 실행: python experiments/20261118-k-compromise/run.py  (KOSMOS 루트에서)
기존 k=100 데이터에서 k를 줄였을 때 결론 유지 여부: 재표본 / L1 축소 추정 / 적응형 k."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl  # noqa: E402

KS = [5, 10, 15, 20, 30, 40, 50]
KS_SHRINK = [5, 10, 20, 30]
ADAPT_TARGETS = [20, 30, 50]
K0 = 10
TAUS = [0, 2, 5, 10, 20]
REPS = 10
BETAS = np.round(np.arange(0.05, 1.5001, 0.01), 3)
MODELS = ["exaone", "deepseek", "haiku"]

df = sl.load()                                   # test 제외
df = df.sort_values(sl.KEY + ["opt_idx"], kind="stable").reset_index(drop=True)
grp = df.groupby(sl.KEY, sort=False).indices
counts = df["count"].to_numpy().astype(float)


def with_counts(c):
    out = df.copy()
    out["count"] = c
    return out


def subsample(k, rng):
    new = counts.copy()
    for idx in grp.values():
        c = counts[idx].astype(int)
        if c.sum() > k:
            new[idx] = rng.multivariate_hypergeometric(c, k)
    return new


def adaptive(target, rng):
    """처음 K0개가 모두 같으면 멈추고 개수를 ×target/K0로 늘린다. 반환: 개수, 프롬프트당 호출 수 배열."""
    new = counts.copy()
    calls = []
    for idx in grp.values():
        c = counts[idx].astype(int)
        n = c.sum()
        seq = rng.permutation(np.repeat(np.arange(len(c)), c))
        first = seq[:K0]
        if len(first) == K0 and (first == first[0]).all():
            v = np.bincount(first, minlength=len(c)).astype(float) * target / K0
            calls.append(K0)
        else:
            take = seq[:min(target, n)]
            v = np.bincount(take, minlength=len(c)).astype(float)
            calls.append(len(take))
        new[idx] = v
    return new, np.array(calls)


def shrink(c, tau):
    """L1 행만: 개수 + τ × 같은 (model, var) L1 10셀 합산 비율."""
    if tau == 0:
        return c
    d = with_counts(c)
    l1 = d["level"] == "L1"
    pooled = d[l1].groupby(["model", "var", "opt_idx"])["count"].transform("sum")
    tot = d[l1].groupby(["model", "var"])["count"].transform("sum")
    out = c.copy()
    out[l1.to_numpy()] = c[l1.to_numpy()] + tau * (pooled / tot.clip(lower=1)).to_numpy()
    return out


# 페르소나 신호: L1 서열 문항에서 셀별 평균 보기 위치
ORD = (df["level"] == "L1") & (df["scale_type"] == "ordinal_monotone")


def cell_means(c):
    d = with_counts(c)[ORD.to_numpy()]
    w = d["count"]
    m = (d["opt_idx"] * w).groupby([d["model"], d["var"], d["persona"]]).sum() / \
        w.groupby([d["model"], d["var"], d["persona"]]).sum().clip(lower=1e-9)
    return m.unstack("persona")


def pair_diffs(means):
    cols = list(means.columns)
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            out.append((means[cols[i]] - means[cols[j]]).rename(f"{cols[i]}|{cols[j]}"))
    return pd.concat(out, axis=1)


FULL_DIFF = pair_diffs(cell_means(counts))


def sign_agree(c):
    d = pair_diffs(cell_means(c))
    res = {}
    for m in MODELS:
        f, x = FULL_DIFF.loc[m].stack(), d.loc[m].stack()
        keep = f.abs() >= 0.1
        res[m] = float((np.sign(f[keep]) == np.sign(x[keep])).mean())
    return res


def evaluate(c, beta):
    val = with_counts(c)
    val = val[val["split"] == "validation"]
    return sl.summary(val, beta)


def fit(c):
    return sl.fit_beta(with_counts(c), betas=BETAS)


rows = []


def record(method, k, rep, c, beta, extra=None):
    s = evaluate(c, beta)
    agree = sign_agree(c)
    for m in MODELS:
        for lv in sl.LEVELS:
            rows.append(dict(method=method, k=k, rep=rep, model=m, level=lv, tvd=s.loc[m, lv],
                             beta=beta[(m, lv)], sign_agree=agree[m], **(extra or {})))


# 기준선: 전체 데이터
beta_full = fit(counts)
record("full", 100, 0, counts, beta_full, {"calls_mean": 100.0})
print("full 완료", flush=True)

for ki, k in enumerate(KS):
    for rep in range(REPS):
        rng = np.random.default_rng(10_000 + 100 * ki + rep)
        c = subsample(k, rng)
        record("raw", k, rep, c, fit(c), {"calls_mean": float(k)})
        if k in KS_SHRINK:
            best = None
            for tau in TAUS:
                cs = shrink(c, tau)
                b = fit(cs)
                cal = with_counts(cs)
                v = sl.summary(cal[cal["split"] == "calibration"], b)["L1"].mean()
                if best is None or v < best[0] - 1e-12:
                    best = (v, tau, cs, b)
            record("shrink", k, rep, best[2], best[3], {"calls_mean": float(k), "tau": best[1]})
    print("k =", k, "완료", flush=True)

for ti, t in enumerate(ADAPT_TARGETS):
    for rep in range(REPS):
        rng = np.random.default_rng(20_000 + 100 * ti + rep)
        c, calls = adaptive(t, rng)
        km = np.array([key[0] for key in grp.keys()])
        b = fit(c)
        s_ = evaluate(c, b)
        agree = sign_agree(c)
        for m in MODELS:
            cm = calls[km == m]
            for lv in sl.LEVELS:
                rows.append(dict(method="adaptive", k=t, rep=rep, model=m, level=lv, tvd=s_.loc[m, lv],
                                 beta=b[(m, lv)], sign_agree=agree[m], calls_mean=float(cm.mean()),
                                 stopped_share=float((cm == K0).mean())))
    print("adaptive target =", t, "완료", flush=True)

r = pd.DataFrame(rows)
full = r[r.method == "full"].set_index(["model", "level"])["tvd"]
r["d_tvd"] = r.apply(lambda x: x.tvd - full[(x.model, x.level)], axis=1)

# 기준 3: 모델 비교 부호 (L1)
l1 = r[r.level == "L1"].pivot_table(index=["method", "k", "rep"], columns="model", values="tvd")
fl1 = full.xs("L1", level="level")
cmp_rows = []
for (a, b) in [("exaone", "deepseek"), ("exaone", "haiku"), ("deepseek", "haiku")]:
    same = np.sign(l1[a] - l1[b]) == np.sign(fl1[a] - fl1[b])
    g = same.groupby(level=["method", "k"]).mean().rename("sign_same_share").reset_index()
    g["pair"] = f"{a}-{b}"
    cmp_rows.append(g)
cmp = pd.concat(cmp_rows)

summ = r.groupby(["method", "k", "model", "level"]).agg(
    tvd=("tvd", "mean"), tvd_sd=("tvd", "std"), d_tvd=("d_tvd", "mean"), beta=("beta", "mean"),
    sign_agree=("sign_agree", "mean"), calls_mean=("calls_mean", "mean")).reset_index()

# 판정
verdict = []
for (meth, k), g in summ[summ.method != "full"].groupby(["method", "k"]):
    g01 = g[g.level.isin(["L0", "L1"])]
    c1 = bool((g01.d_tvd < 0.01).all())
    c2 = bool((g.groupby("model").sign_agree.first() >= 0.9).all())
    cc = cmp[(cmp.method == meth) & (cmp.k == k)]
    c3 = bool((cc[cc.pair.str.startswith("exaone")].sign_same_share >= 0.9).all())
    verdict.append(dict(method=meth, k=k, calls_mean=float(g.calls_mean.mean()),
                        worst_d_tvd_L0L1=float(g01.d_tvd.max()),
                        min_sign_agree=float(g.groupby("model").sign_agree.first().min()),
                        min_model_cmp=float(cc[cc.pair.str.startswith("exaone")].sign_same_share.min()),
                        c1_tvd=c1, c2_persona=c2, c3_model_cmp=c3, pass_all=c1 and c2 and c3))
verdict = pd.DataFrame(verdict).sort_values(["method", "k"])
if "stopped_share" in r:
    stop = r[r.method == "adaptive"].groupby(["k", "model"]).stopped_share.mean().unstack()
else:
    stop = pd.DataFrame()

pd.set_option("display.width", 200)
print(verdict.round(4).to_string(index=False))
print(summ[summ.level.isin(["L0", "L1"])].pivot_table(index=["method", "k"], columns=["model", "level"],
                                                      values="d_tvd").round(4))
print("적응형: 모델별 평균 호출 수 / 멈춘 비율")
print(r[(r.method == "adaptive") & (r.level == "L1")].groupby(["k", "model"])[["calls_mean", "stopped_share"]].mean().round(3))

tables = {"summary": summ.set_index(["method", "k", "model", "level"]),
          "verdict": verdict.set_index(["method", "k"]),
          "model_compare_sign": cmp.set_index(["method", "k", "pair"]),
          "reps": r.set_index(["method", "k", "rep", "model", "level"])}
run = sl.save_run(HERE / "out", params={"ks": KS, "ks_shrink": KS_SHRINK, "taus": TAUS, "adapt_targets": ADAPT_TARGETS,
                                       "k0": K0, "reps": REPS, "beta_full": beta_full, "alpha": sl.ALPHA},
                  tables=tables)
print(f"\n기록: {HERE / 'out/run.json'} (commit {run['git_commit']}, dirty={run['git_dirty']})")
