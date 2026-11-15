"""20261115-k-curve — 실행: python experiments/20261115-k-curve/run.py  (KOSMOS 루트에서)
count 벡터에서 비복원 추출로 k를 줄여 보정 후 validation TVD의 k 의존성을 잰다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research/design_v2_20261112/dataset"))
import silicon as sl  # noqa: E402

KS = [5, 10, 20, 30, 50, 70, 90]
REPS = 5
BETAS = np.round(np.arange(0.05, 1.5001, 0.01), 3)
df = sl.load()                                   # test 제외
df = df.sort_values(sl.KEY + ["opt_idx"], kind="stable").reset_index(drop=True)
grp = df.groupby(sl.KEY, sort=False).indices     # 키 → 행 위치 배열
counts = df["count"].to_numpy()


def subsample(k, rng):
    new = counts.copy()
    for idx in grp.values():
        c = counts[idx]
        n = int(c.sum())
        if n > k:
            new[idx] = rng.multivariate_hypergeometric(c, k)
    out = df.copy()
    out["count"] = new
    return out


rows = []
for ki, k in enumerate(KS):
    for rep in range(REPS):
        rng = np.random.default_rng(1000 * ki + rep)
        d = subsample(k, rng)
        beta = sl.fit_beta(d, betas=BETAS)
        val = d[d["split"] == "validation"]
        s = sl.summary(val, beta)
        for model in s.index:
            for lv in sl.LEVELS:
                rows.append(dict(k=k, rep=rep, model=model, level=lv,
                                 tvd=s.loc[model, lv], beta=beta[(model, lv)]))
    print("k =", k, "완료", flush=True)

# 전체 데이터(실제 k) 기준
beta_full = sl.fit_beta(df, betas=BETAS)
full = sl.summary(df[df["split"] == "validation"], beta_full)
r = pd.DataFrame(rows)
curve = r.groupby(["model", "level", "k"]).agg(tvd_mean=("tvd", "mean"), tvd_sd=("tvd", "std"),
                                              beta_mean=("beta", "mean")).reset_index()
uni = sl.uniform_baseline(df[df["split"] == "validation"]).stack().rename("uniform").reset_index()
uni.columns = ["model", "level", "uniform"]
tables = {"k_curve": curve.set_index(["model", "level", "k"]),
          "full_data_tvd": full,
          "uniform_baseline": uni.set_index(["model", "level"])}
print(curve.pivot_table(index=["model", "level"], columns="k", values="tvd_mean").round(3))

# 그림
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
names = {"exaone": "K-EXAONE 2.5", "deepseek": "DeepSeek V4.1 Flash", "haiku": "Claude Haiku 5.5"}
colors = {"exaone": "#d95f02", "deepseek": "#1b9e77", "haiku": "#7570b3"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, lv in zip(axes, sl.LEVELS):
    for m in names:
        c = curve[(curve.model == m) & (curve.level == lv)]
        ax.plot(c.k, c.tvd_mean, "-o", color=colors[m], label=names[m], ms=4)
        ax.fill_between(c.k, c.tvd_mean - c.tvd_sd, c.tvd_mean + c.tvd_sd, color=colors[m], alpha=0.2)
    u = uni[(uni.level == lv)].uniform.mean()
    ax.axhline(u, color="gray", ls="--", lw=1, label="균등 분포")
    ax.set_title({"c0": "c0 (페르소나 없음)", "L0": "L0 (한국 성인)", "L1": "L1 (성별×연령)"}[lv])
    ax.set_xlabel("프롬프트당 표본 수 k")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("보정 후 validation TVD (낮을수록 좋음)")
axes[0].legend(fontsize=8)
fig.suptitle("k에 따른 보정 후 TVD (k마다 beta 재선택, 5회 반복 평균±sd)")
fig.tight_layout()
(HERE / "out").mkdir(exist_ok=True)
fig.savefig(HERE / "out/k_curve.png", dpi=130)

run = sl.save_run(HERE / "out", params={"ks": KS, "reps": REPS, "beta_full": beta_full, "alpha": sl.ALPHA},
                  tables=tables)
print(f"\n기록: {HERE / 'out/run.json'} (commit {run['git_commit']}, dirty={run['git_dirty']})")
