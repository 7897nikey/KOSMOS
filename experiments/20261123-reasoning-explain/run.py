"""20261123-reasoning-explain — 추론 low가 페르소나 반응을 키우는 방식을 셀 간 분산비·셀 편차 정렬로 분해 (k=50).
실행 (KOSMOS 루트): uv run --no-project --python 3.12 --with pandas --with numpy python experiments/20261123-reasoning-explain/run.py

대상: test를 뺀 순서형·이진 문항. 점수 s = 응답 코드를 0~1로 바꾼 평균. 10셀 가중치 pop_share.
"""
import csv
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS = DESIGN / "prompts_v2.csv"
MANIFEST = DESIGN / "item_manifest_v2.csv"
CELLS = DESIGN / "cells.csv"
RAW = {"끔": ROOT / "raw/v2/deepseek_eco_p1.jsonl", "low": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl"}
K, B, NULL_REPS = 50, 2000, 200
rng = np.random.default_rng(20261123)
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# ── 1. 데이터 ───────────────────────────────────────────
d = np.load(NPZ, allow_pickle=True)
H, NEFF, CODES, SPLIT = d["H"], d["NEFF"], d["CODES"], d["split"]
PERSONAS, VARS = list(d["personas"]), list(d["vars"])
MASK = ~np.isnan(CODES)
NI = len(VARS)
items = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == VARS
ORD = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
EV = SPLIT != "test"
SEL = ORD & EV
L1J = list(range(2, 12))
cells = {r["cell_id"]: r for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
W = np.array([float(cells[PERSONAS[j]]["pop_share"]) for j in L1J])
W /= W.sum()
AGE = {PERSONAS[j]: cells[PERSONAS[j]]["age_wmedian"] for j in L1J}
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}
NORM = np.zeros(CODES.shape)                                   # 코드 → 0~1
for i in range(NI):
    c = CODES[i][MASK[i]]
    NORM[i, :MASK[i].sum()] = (c - c.min()) / max(c.max() - c.min(), 1)
GENDER = re.compile(r"female|male|woman|\bman\b|男|女|남자|여자|남성|여성", re.I)


def lang_of(t):
    h = t[:50]
    if re.search(r"[가-힣]", h):
        return "한국어"
    if re.search(r"[一-鿿]", h):
        return "중국어"
    return "영어"


def load(path, keep_text=False):
    """개수 부분 출처: experiments/20261118-reasoning-effect/run.py load_counts (표본 0~49, 중복·api_error 제외)."""
    C = np.zeros((len(PERSONAS), NI, CODES.shape[1]))
    recs, seen = [], set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K or key in seen or r["probe_id"] not in probes:
            continue
        seen.add(key)
        p = probes[r["probe_id"]]
        j, i = PERSONAS.index(p["cell_id"]), VARS.index(p["var"])
        if r["valid"]:
            k_ = list(CODES[i]).index(float(r["parsed"]))
            C[j, i, k_] += 1
            if keep_text and j >= 2:
                t = r.get("reasoning_text") or ""
                recs.append(dict(j=j, i=i, k=k_, mention=bool(AGE[p["cell_id"]] in t or GENDER.search(t)),
                                 lang=lang_of(t), rtok=r.get("reasoning_tokens", 0)))
    return C, recs


C_OFF, _ = load(RAW["끔"])
C_LOW, RECS = load(RAW["low"], keep_text=True)
log(f"순서형·이진 문항(test 제외): {int(SEL.sum())}개, 추론 low 10셀 유효 응답 {len(RECS):,}건")


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def stats(P, n):
    """P [셀, 문항, 보기] 비율, n [셀, 문항] 표본수 → 점수 s, 잡음 분산 σ² [셀, 문항]."""
    s = (P * NORM).sum(-1)
    var = (P * (NORM - s[..., None]) ** 2 * MASK).sum(-1)
    return s, var / np.maximum(n, 1)


def parts(s, sig2):
    """문항별: 잡음 뺀 셀 간 분산, 편차 d [셀, 문항]."""
    sbar = (W[:, None] * s).sum(0)
    dv = s - sbar
    V = (W[:, None] * dv ** 2).sum(0) - (W * (1 - W))[:, None].__mul__(sig2).sum(0)
    return V, dv


sH, sigH = stats(H[L1J], NEFF[L1J])
VH, dH = parts(sH, sigH)
M = {}
for name, C in (("끔", C_OFF), ("low", C_LOW)):
    s, sig = stats(share(C[L1J]), C[L1J].sum(-1))
    V, dm = parts(s, sig)
    M[name] = dict(s=s, V=V, d=dm, num=(W[:, None] * dm * dH).sum(0), dm2=(W[:, None] * dm ** 2).sum(0))
DH2 = (W[:, None] * dH ** 2).sum(0)
IDX = np.where(SEL)[0]
BIX = IDX[rng.integers(0, len(IDX), (B, len(IDX)))]


def agg(name, ix):
    m = M[name]
    return dict(분산비=m["V"][ix].sum(-1) / VH[ix].sum(-1),
                정렬=m["num"][ix].sum(-1) / np.sqrt(m["dm2"][ix].sum(-1) * DH2[ix].sum(-1)),
                기울기=m["num"][ix].sum(-1) / DH2[ix].sum(-1))


A_pt = {n: agg(n, IDX) for n in M}
A_bs = {n: agg(n, BIX) for n in M}


def summarize(key):
    d_ = A_bs["low"][key] - A_bs["끔"][key]
    p = max(2 * min((d_ <= 0).mean(), (d_ >= 0).mean()), 1 / B)
    return dict(지표=key, 추론_끔=float(A_pt["끔"][key]), 끔_하한=float(np.percentile(A_bs["끔"][key], 2.5)), 끔_상한=float(np.percentile(A_bs["끔"][key], 97.5)),
                추론_low=float(A_pt["low"][key]), low_하한=float(np.percentile(A_bs["low"][key], 2.5)), low_상한=float(np.percentile(A_bs["low"][key], 97.5)),
                차이=float(A_pt["low"][key] - A_pt["끔"][key]), 하한=float(np.percentile(d_, 2.5)), 상한=float(np.percentile(d_, 97.5)), p=float(min(p, 1)))


pr = pd.DataFrame([summarize("분산비"), summarize("정렬")])
order = np.argsort(pr["p"].to_numpy())
adj, run = np.empty(len(pr)), 0.0
for r_, i in enumerate(order):
    run = max(run, min(1.0, (len(pr) - r_) * pr["p"].iloc[i]))
    adj[i] = run
pr["p_Holm"] = adj
pr["있음"] = pr["p_Holm"] < 0.05
pr.to_csv(out / "primary.csv", index=False, encoding="utf-8-sig")
log("\n[주 지표 (10셀, 순서형·이진 문항, low − 끔, Holm 2개)]")
log(pr.round(4).to_string(index=False))
log(f"참고: 사람 셀 간 분산(잡음 뺀 값) 합 {VH[IDX].sum():.4f}, 잡음 빼기 전 {(W[:, None] * dH ** 2).sum(0)[IDX].sum():.4f}")

ex = pd.DataFrame([summarize("기울기")])

# ── 2. 이동 분해 ─────────────────────────────────────────
delta = M["low"]["s"] - M["끔"]["s"]
common = (W[:, None] * delta).sum(0)
spec = (W[:, None] * (delta - common) ** 2).sum(0)
pooled = share(C_OFF[L1J] + C_LOW[L1J])
n_off, n_low = C_OFF[L1J].sum(-1).astype(int), C_LOW[L1J].sum(-1).astype(int)
nc, ns = np.zeros(NI), np.zeros(NI)
for i in IDX:
    k_ = MASK[i].sum()
    so = np.stack([rng.multinomial(n_off[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(n_off[c, i], 1) for c in range(10)])
    sl = np.stack([rng.multinomial(n_low[c, i], pooled[c, i, :k_], size=NULL_REPS) @ NORM[i, :k_] / max(n_low[c, i], 1) for c in range(10)])
    dl = sl - so                                                # [셀, 반복]
    cm = (W[:, None] * dl).sum(0)
    nc[i] = (cm ** 2).mean()
    ns[i] = (W[:, None] * (dl - cm) ** 2).sum(0).mean()
s_pop_h = (W[:, None] * sH).sum(0)
s_off_bar, s_low_bar = (W[:, None] * M["끔"]["s"]).sum(0), (W[:, None] * M["low"]["s"]).sum(0)
toward = np.abs(s_low_bar - s_pop_h) - np.abs(s_off_bar - s_pop_h)
sL0 = {n: stats(share(C[[1]]), C[[1]].sum(-1))[0][0] for n, C in (("끔", C_OFF), ("low", C_LOW))}
sL0h = stats(H[[1]], NEFF[[1]])[0][0]
toward_L0 = np.abs(sL0["low"] - sL0h) - np.abs(sL0["끔"] - sL0h)


def bmean(x):
    v = x[BIX].mean(1)
    return float(x[IDX].mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


drow = []
for lab, x in [("공통 이동 제곱", common ** 2), ("공통 이동 제곱 잡음", nc), ("공통 이동 제곱 − 잡음", common ** 2 - nc),
               ("셀 고유 이동 제곱합", spec), ("셀 고유 이동 잡음", ns), ("셀 고유 이동 − 잡음", spec - ns),
               ("10셀 평균이 사람 평균에 가까워진 정도(음수=가까워짐)", toward), ("한국 성인 점수가 사람 평균에 가까워진 정도(음수=가까워짐)", toward_L0),
               ("셀 고유 이동이 사람 셀 편차와 같은 방향(공분산)", (W[:, None] * (delta - common) * dH).sum(0))]:
    m, lo, hi = bmean(x)
    drow.append(dict(항목=lab, 평균=m, 하한=lo, 상한=hi))
dd = pd.DataFrame(drow)
dd.to_csv(out / "decomp.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 추론 끔 → low 점수 이동 분해 (문항 평균, 10셀)]")
log(dd.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

# ── 3. 추론 본문 ─────────────────────────────────────────
rd = pd.DataFrame(RECS)
rd["s"] = NORM[rd["i"], rd["k"]]
log("\n[탐색: 추론 본문 (추론 low, 10셀, 유효 응답)]")
log("언어 구성:", rd["lang"].value_counts(normalize=True).round(3).to_dict())
log("언어별 페르소나 언급률:", rd.groupby("lang")["mention"].mean().round(3).to_dict())
log("전체 언급률:", round(rd["mention"].mean(), 3), "| 추론 토큰 평균 언급/비언급:", rd.groupby("mention")["rtok"].mean().round(1).to_dict())
mr = rd.groupby("i")["mention"].mean().reindex(range(NI))
dV = M["low"]["V"] - M["끔"]["V"]
ok = np.array([i for i in IDX if not np.isnan(mr[i])])
rk = lambda x: pd.Series(x).rank().to_numpy()


def spear(ix):
    return np.corrcoef(rk(mr.to_numpy()[ix]), rk(dV[ix]))[0, 1]


sb = np.array([spear(ok[rng.integers(0, len(ok), len(ok))]) for _ in range(B)])
trow = [dict(항목="문항별 언급률 ↔ 셀 간 분산 증가 (스피어만)", 값=float(spear(ok)), 하한=float(np.percentile(sb, 2.5)), 상한=float(np.percentile(sb, 97.5)), n=len(ok))]
g = rd[np.isin(rd["i"], IDX)].groupby(["j", "i", "mention"])["s"].agg(["mean", "size"]).unstack("mention")
g = g[(g[("size", True)] >= 3) & (g[("size", False)] >= 3)].dropna()
diff = (g[("mean", True)] - g[("mean", False)]).to_numpy()
dh = np.array([dH[j - 2, i] for j, i in g.index])
hit = np.where(diff == 0, 0.5, (np.sign(diff) == np.sign(dh)).astype(float))
hit[dh == 0] = np.nan
gi = np.array([i for _, i in g.index])
uniq = np.unique(gi[~np.isnan(hit)])
hb = []
for _ in range(B):
    pick = rng.choice(uniq, len(uniq))
    hb.append(np.nanmean(np.concatenate([hit[gi == u] for u in pick])))
trow.append(dict(항목="같은 프롬프트 안 언급−비언급 점수 차이가 사람 셀 편차와 같은 방향인 비율", 값=float(np.nanmean(hit)),
                 하한=float(np.percentile(hb, 2.5)), 상한=float(np.percentile(hb, 97.5)), n=int((~np.isnan(hit)).sum())))
td = pd.DataFrame(trow)
td.to_csv(out / "reasoning_text.csv", index=False, encoding="utf-8-sig")
log(td.round(4).to_string(index=False))
ex.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 기울기 (사람 셀 차이 1당 모델 셀 차이)]")
log(ex.round(4).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for k, p in RAW.items()},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "seed": 20261123},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
