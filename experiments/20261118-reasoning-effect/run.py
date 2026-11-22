"""20261118-reasoning-effect (E0) — DeepSeek Eco 추론 끔 vs low, k=50.
실행 (KOSMOS 루트): python experiments/20261118-reasoning-effect/run.py

1. 원응답 → 개수 C[페르소나, 문항, 보기] (표본 0~49, 중복·api_error 제외)
2. 조건별 지표 (D1~D5)
3. low − 끔, 같은 문항끼리 짝지은 부트스트랩 95% 구간
4. 판정표 + 참고: 끔 Eco − 끔 Ultrafast
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
PROMPTS = ROOT / "research/design_v2_20261112/prompts_v2.csv"
MANIFEST = ROOT / "research/design_v2_20261112/item_manifest_v2.csv"
RAW = {"끔": ROOT / "raw/v2/deepseek_eco_p1.jsonl",
       "low": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl",
       "끔 Ultrafast": ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl"}
K = 50
B = 2000
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
rng = np.random.default_rng(20261119)

# ── 1. 데이터 ───────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
H, NEFF, CODES, SPLIT = d["H"], d["NEFF"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
L1 = slice(2, 12)
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORDINAL = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
EV, CAL, VAL = SPLIT != "test", SPLIT == "calibration", SPLIT == "validation"
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}


def load_counts(path):
    C = np.zeros((len(PERSONAS), len(VARS), CODES.shape[1]))
    n = np.zeros((len(PERSONAS), len(VARS)))
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in probes:
            continue
        seen.add(key)
        p = probes[r["probe_id"]]
        j, i = PERSONAS.index(p["cell_id"]), VARS.index(p["var"])
        n[j, i] += 1
        if r["valid"]:
            C[j, i, list(CODES[i]).index(float(r["parsed"]))] += 1
    return C, n


# ── 2. 지표 ─────────────────────────────────────────────
def prob(X, beta=1.0):
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def score(P):
    return (P * np.nan_to_num(CODES)).sum(-1)


def cell_pairs(x):
    return x[..., :, None, :] - x[..., None, :, :]


top = lambda x: np.where(MASK, x, -np.inf).argmax(-1)

# 사람 쪽 유의한 셀 쌍 (k-keep-findings와 같은 규칙·시드)
hrng = np.random.default_rng(20261112)
Hb = np.zeros((1000, *H.shape))
for j in range(H.shape[0]):
    for i in range(H.shape[1]):
        k_ = MASK[i].sum()
        Hb[:, j, i, :k_] = hrng.multinomial(NEFF[j, i], H[j, i, :k_], size=1000) / NEFF[j, i]
DH_CELL = cell_pairs(score(H[L1]))
lo_, hi_ = np.percentile(cell_pairs(score(Hb[:, L1])), [2.5, 97.5], axis=0)
CELL_PAIRS = np.triu(np.ones((10, 10), bool), 1)[..., None] & ((lo_ > 0) | (hi_ < 0)) & (DH_CELL != 0) & (ORDINAL & EV)
DH_OPT = H[L1][..., :, None] - H[L1][..., None, :]
OPT_PAIRS = (np.triu(np.ones((7, 7), bool), 1)[None, None] & MASK[None, :, :, None] & MASK[None, :, None, :]
             & EV[None, :, None, None] & (DH_OPT != 0))


def hits(dm, dh, valid, item_axis):
    """방향이 같으면 1, 모델 쪽 차이 0이면 0.5. 문항별 (맞힌 수, 쌍 수)."""
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0)) * valid
    axes = tuple(a for a in range(valid.ndim) if a != item_axis)
    return s.sum(axes), valid.sum(axes).astype(float)


def per_item(C):
    """문항별 값: D1 L0·L1 보정 TVD, D2 c0→L0 변화, D3·D4 (맞힌 수, 쌍 수), D5 최빈값 일치."""
    lv = [C[0][None], C[1][None], C[L1]]                                  # 단계별 [셀, 문항, 보기]
    hv = [H[0][None], H[1][None], H[L1]]
    t_cal = []
    for c, h in zip(lv, hv):
        g = tvd(prob(c[None], BETAS[:, None, None, None]), h).mean(1)     # [β, 문항]
        t_cal.append(g[g[:, CAL].mean(1).argmin()])
    t1 = tvd(prob(C), H)
    R = share(C)
    return {
        "D1 보정 TVD L0": t_cal[1], "D1 보정 TVD L1": t_cal[2],
        "D2 c0→L0 변화": t1[1] - t1[0],
        "D3 셀 간 AUC": hits(cell_pairs(score(R[L1])), DH_CELL, CELL_PAIRS, 2),
        "D4 보기 순서 AUC": hits(R[L1][..., :, None] - R[L1][..., None, :], DH_OPT, OPT_PAIRS, 1),
        "D5 최빈값 일치 L0": (top(R[1]) == top(H[1])).astype(float),
        "D5 최빈값 일치 L1": (top(R[L1]) == top(H[L1])).mean(0),
    }


def paired(a, b, sel):
    """b − a, 같은 문항 부트스트랩. 값이 (맞힌 수, 쌍 수)면 AUC로."""
    idx = np.where(sel)[0]
    boot = rng.integers(0, len(idx), (B, len(idx)))
    if isinstance(a, tuple):
        f = lambda x, ix: x[0][ix].sum(-1) / np.maximum(x[1][ix].sum(-1), 1)
        va, vb = f(a, idx), f(b, idx)
        diff = f(b, idx[boot]) - f(a, idx[boot])
    else:
        va, vb = a[idx].mean(), b[idx].mean()
        diff = b[idx[boot]].mean(1) - a[idx[boot]].mean(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(va), float(vb), float(vb - va), float(lo), float(hi)


SCOPE = {"D1 보정 TVD L0": VAL, "D1 보정 TVD L1": VAL}

# ── 3. 비교 ─────────────────────────────────────────────
counts = {name: load_counts(p) for name, p in RAW.items()}
for name, (C, n) in counts.items():
    print(f"{name}: 프롬프트당 표본 수 최소 {int(n[n > 0].min())}, 최대 {int(n.max())}, 빈 프롬프트 {int((n == 0).sum())}, 유효율 {C.sum() / n.sum():.3f}")
vals = {name: per_item(C) for name, (C, _) in counts.items()}

rows = []
for (a, b, kind) in [("끔", "low", "판정"), ("끔 Ultrafast", "끔", "참고: 서빙판 차이")]:
    for m in vals[a]:
        va, vb, diff, lo, hi = paired(vals[a][m], vals[b][m], SCOPE.get(m, EV))
        rows.append(dict(비교=f"{b} − {a}", 용도=kind, 지표=m, **{a: va, b: vb}, 차이=diff, 하한=lo, 상한=hi,
                         유의=bool(lo > 0 or hi < 0)))
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
effect = bool(res[res.용도 == "판정"].유의.any())
print("\n추론 효과 있음 (E1+ 실행):" if effect else "\n추론 효과 없음 (E1+ 생략):",
      list(res[(res.용도 == "판정") & res.유의].지표))

out = HERE / "out"
out.mkdir(exist_ok=True)
res.to_csv(out / "diffs.csv", index=False)
git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for k, p in RAW.items()},
           "params": {"k": K, "bootstrap": B, "beta_grid": [0.05, 1.5, 0.005]}, "reasoning_effect": effect,
           "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
