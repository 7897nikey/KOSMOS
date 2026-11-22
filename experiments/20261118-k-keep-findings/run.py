"""20261118-k-keep-findings — k=100의 결론이 줄인 k에서도 유지되는가.
실행 (KOSMOS 루트): python experiments/20261118-k-keep-findings/run.py

1. 데이터: 세 모델의 k=100 개수와 사람 정답지 (counts_k100.npz)
2. 결론 F1~F5를 계산하는 함수 (유의 판단 = 문항 부트스트랩 95% 구간)
3. k=100에서 각 결론의 판단을 정답으로 저장
4. k마다 20회: 원응답 100개에서 k개 비복원 추출 → 같은 판단이 나오는 비율
5. 비율 ≥ 0.9인 결론이 15개 모두인 가장 작은 k
"""
import csv
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NPZ = ROOT / "research/design_v2_20261112/analysis/out_final_k100/counts_k100.npz"
MANIFEST = ROOT / "research/design_v2_20261112/item_manifest_v2.csv"
KS = list(range(5, 101, 5))
REPS = 20
B = 2000                       # 문항 부트스트랩 횟수
PASS = 0.9
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
rng = np.random.default_rng(20261118)

# ── 1. 데이터 ───────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
C, NTOT, H, NEFF, CODES = d["C"], d["NTOT"], d["H"], d["NEFF"], d["CODES"]
MODELS, SPLIT = list(d["models"]), d["split"]
MASK = ~np.isnan(CODES)                                   # [문항, 보기] 실제 있는 보기
L1 = slice(2, 12)                                         # 페르소나 축: c0, L0, L1 10셀
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == list(d["vars"]), "문항 순서가 다르다"
ORDINAL = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
EV, CAL, VAL = SPLIT != "test", SPLIT == "calibration", SPLIT == "validation"


# ── 2. 지표와 결론 ───────────────────────────────────────
def prob(X, beta=1.0):
    """라플라스 평활(+1) 후 β 제곱. 마지막 축이 보기."""
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def score(P):
    """서열 문항의 평균 점수 (보기 코드 가중평균)."""
    return (P * np.nan_to_num(CODES)).sum(-1)


def cell_pairs(x):
    """[..., 셀, 문항] → 모든 셀 쌍의 차이 [..., 셀a, 셀b, 문항]."""
    return x[..., :, None, :] - x[..., None, :, :]


def ci_mean(x):
    """문항별 값 x의 평균과 부트스트랩 95% 구간."""
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def ci_auc(hit, n):
    """문항별 맞힌 수 hit, 쌍 수 n → AUC와 부트스트랩 95% 구간."""
    keep = n > 0
    hit, n = hit[keep], n[keep]
    idx = rng.integers(0, len(n), (B, len(n)))
    b = hit[idx].sum(1) / n[idx].sum(1)
    return hit.sum() / n.sum(), *np.percentile(b, [2.5, 97.5])


def auc_parts(dm, dh, valid):
    """방향이 같으면 1, 모델 쪽 차이가 0이면 0.5. 문항(마지막 축)별 합."""
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    axes = tuple(range(valid.ndim - 1))
    return (s * valid).sum(axes), valid.sum(axes).astype(float)


# 사람 쪽 유의한 셀 쌍 (RADIUS): 사람 응답을 NEFF명으로 1,000회 재표집해 점수 차이 구간이 0을 안 포함하는 쌍
hrng = np.random.default_rng(20261112)
Hb = np.zeros((1000, *H.shape))
for j in range(H.shape[0]):
    for i in range(H.shape[1]):
        k_ = MASK[i].sum()
        Hb[:, j, i, :k_] = hrng.multinomial(NEFF[j, i], H[j, i, :k_], size=1000) / NEFF[j, i]
DH_CELL = cell_pairs(score(H[L1]))                                         # [셀a, 셀b, 문항]
lo, hi = np.percentile(cell_pairs(score(Hb[:, L1])), [2.5, 97.5], axis=0)
SIG = (lo > 0) | (hi < 0)
UP = np.triu(np.ones((10, 10), bool), 1)[..., None]
CELL_PAIRS = UP & SIG & (DH_CELL != 0) & (ORDINAL & EV)[None, None, :]
# 셀 안 보기 쌍: [셀, 문항, 보기a, 보기b]
DH_OPT = H[L1][..., :, None] - H[L1][..., None, :]
OPT_PAIRS = (np.triu(np.ones((7, 7), bool), 1)[None, None] & MASK[None, :, :, None] & MASK[None, :, None, :]
             & EV[None, :, None, None] & (DH_OPT != 0))


def findings(Cm):
    """한 모델의 개수 [페르소나, 문항, 보기] → 결론 5개의 (추정, 하한, 상한, 영가설 값)."""
    t = tvd(prob(Cm), H)                                                   # β=1, [페르소나, 문항]
    out = {
        "F1 c0→L0": (*ci_mean((t[1] - t[0])[EV]), 0.0),
        "F2 L0→L1": (*ci_mean((t[L1].mean(0) - t[1])[EV]), 0.0),
    }
    R = share(Cm)
    dm = cell_pairs(score(R[L1]))
    out["F3 셀 간 AUC"] = (*ci_auc(*auc_parts(dm, DH_CELL, CELL_PAIRS)), 0.5)
    dmo = R[L1][..., :, None] - R[L1][..., None, :]
    hit, n = auc_parts(dmo.transpose(0, 2, 3, 1), DH_OPT.transpose(0, 2, 3, 1), OPT_PAIRS.transpose(0, 2, 3, 1))
    out["F4 보기 순서 AUC"] = (*ci_auc(hit, n), 0.5)
    # F5: calibration에서 β* → validation에서 TVD(β*) − TVD(균등), L1
    tb = tvd(prob(Cm[None, L1], BETAS[:, None, None, None]), H[L1]).mean(1)  # [β, 문항]
    bstar = BETAS[tb[:, CAL].mean(1).argmin()]
    unif = tvd(MASK / MASK.sum(1, keepdims=True), H[L1]).mean(0)
    out["F5 보정>균등"] = (*ci_mean((tb[BETAS == bstar][0] - unif)[VAL]), 0.0)
    return out


def verdict(est, lo, hi, null):
    """+1 = 영가설보다 큼(유의), -1 = 작음(유의), 0 = 유의하지 않음."""
    return 1 if lo > null else -1 if hi < null else 0


# ── 3. k=100 판단 ───────────────────────────────────────
rows = []
for a, m in enumerate(MODELS):
    for f, (est, lo_, hi_, null) in findings(C[a]).items():
        rows.append(dict(model=m, finding=f, est=est, lo=lo_, hi=hi_, verdict=verdict(est, lo_, hi_, null)))
ref = pd.DataFrame(rows)
REF = {(r.model, r.finding): r.verdict for r in ref.itertuples()}
print("k=100 판단 (+1/-1 유의, 0 비유의)\n", ref.round(4).to_string(index=False), flush=True)


# ── 4. k별 재표본 ───────────────────────────────────────
def subsample(a, k):
    """모델 a의 프롬프트마다 원응답(유효 보기 개수 + 무효)에서 k개 비복원 추출 → 유효 개수만."""
    out = np.zeros_like(C[a])
    for j in range(C.shape[1]):
        for i in range(C.shape[2]):
            v = np.append(C[a, j, i, MASK[i]], NTOT[a, j, i] - C[a, j, i].sum()).astype(int)
            out[j, i, MASK[i]] = rng.multivariate_hypergeometric(v, min(k, v.sum()))[:-1]
    return out


rows = []
for k in KS:
    for rep in range(REPS if k < 100 else 1):
        for a, m in enumerate(MODELS):
            Ck = C[a] if k == 100 else subsample(a, k)
            for f, (est, lo_, hi_, null) in findings(Ck).items():
                v = verdict(est, lo_, hi_, null)
                rows.append(dict(k=k, rep=rep, model=m, finding=f, est=est, verdict=v, same=v == REF[(m, f)]))
    print("k =", k, "완료", flush=True)
res = pd.DataFrame(rows)

# ── 5. 통과표 ───────────────────────────────────────────
agree = res.groupby(["finding", "model", "k"]).same.mean().unstack("k")
est = res.groupby(["finding", "model", "k"]).est.mean().unstack("k")
passed = (agree >= PASS).all()
k_min = next((k for k in KS if passed[k]), None)
pd.set_option("display.width", 250)
print("\nk=100과 같은 판단이 나온 비율 (20회 중)\n", agree.round(2).to_string())
print("\n모든 결론 통과:", passed.to_dict())
print("\n선택된 k:", k_min)

out = HERE / "out"
out.mkdir(exist_ok=True)
ref.to_csv(out / "verdict_k100.csv", index=False)
agree.to_csv(out / "agreement_by_k.csv")
est.to_csv(out / "estimate_by_k.csv")
res.to_csv(out / "reps.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, m in zip(axes, MODELS):
    for f in agree.index.get_level_values(0).unique():
        ax.plot(KS, agree.loc[(f, m)].values, "-o", ms=3, label=f)
    ax.axhline(PASS, color="gray", ls="--", lw=1)
    if k_min:
        ax.axvline(k_min, color="red", ls=":", lw=1)
    ax.set_title(m)
    ax.set_xlabel("k (프롬프트당 호출 수)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("k=100과 같은 결론이 나온 비율")
axes[0].legend(fontsize=8)
fig.suptitle(f"k를 줄여도 결론이 유지되는가 (점선 {PASS}, 빨간 선 = 선택된 k {k_min})")
fig.tight_layout()
fig.savefig(out / "agreement_by_k.png", dpi=130)

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "data": str(NPZ.relative_to(ROOT)), "data_sha256": hashlib.sha256(NPZ.read_bytes()).hexdigest()[:16],
           "params": {"ks": KS, "reps": REPS, "bootstrap": B, "pass": PASS, "beta_grid": [0.05, 1.5, 0.005]},
           "k_min": k_min, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
