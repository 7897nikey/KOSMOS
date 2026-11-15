"""K-EXAONE 2.5 최종 분석 (k=100) — 설계 v2 본 분석 프레임으로 DeepSeek·Haiku와 비교하고 test를 한 번 연다.

프레임은 exaone_interim_v2.py와 같다 (v2_experiment.ipynb · radius_ties.py와 같은 규칙).
- 정답지: KGSS 2023, FINALWT 가중. 페르소나 12개(c0, L0, L1 10셀).
- 보정: 라플라스 α = 1, p ∝ (c + 1)^β. β 그리드 0.05~1.5(0.005 간격), calibration 148문항 평균 TVD 최소.
- 지표: TVD, 정규화 W1(서열·이진), 최빈값 정확도(엄격 · RADIUS), BPR식 보기 쌍 AUC, 셀 간 쌍별 순위.

중간 분석과 다른 점
- 세 모델 모두 프롬프트당 100표본(표본 번호 0~99 전부). 표본 맞춤이 필요 없다.
- EXAONE 원응답은 merge_check.py가 만든 exaone_p1_merged.jsonl(4명 합본) 하나만 읽는다.
- 순서: (1) calibration에서 EXAONE의 β*·λ*를 고른다 → (2) 개수 해시와 함께
  analysis/out_calib/frozen_beta_exaone_k100.json에 고정한다 → (3) test 49문항을 연다.
  고정 파일이 이미 있으면 다시 고르지 않고 그 값을 쓴다(개수 해시가 다르면 멈춘다). 그래서 다시 돌려도 test는 같은 β로만 평가된다.
- DeepSeek·Haiku는 기존 frozen_beta.json의 β·λ를 그대로 쓴다(이 파일은 읽기만 한다). 재계산 값과 같은지도 기록한다.
- β와 무관한 지표(최빈값·쌍 순위·쏠림 등)는 test / cal+val / 전체 246문항 세 범위로 낸다.

기록 (out_final_k100/)
- run.log: 이 실행의 화면 출력 전체
- run.json: 커밋·작업트리 변경 여부·스크립트 해시·입력 파일 해시·개수 해시·파라미터·버전·test 공개 여부
- counts_k100.npz: 모델 개수 텐서와 사람 정답지. 원응답 없이 같은 분석을 다시 돌릴 수 있다
- *.csv, summary.json: 결과 값

실행 (KOSMOS 루트):
  uv run --no-project --python 3.12 --with pandas --with numpy --with pyarrow \
      python research/design_v2_20261112/analysis/exaone_final_v2.py
"""
from __future__ import annotations

import collections
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "inventory/kgss_clean.parquet").exists())
DESIGN = ROOT / "research/design_v2_20261112"
OUT = DESIGN / "analysis/out_final_k100"
OUT.mkdir(parents=True, exist_ok=True)
PKG = DESIGN / "team_packages/KOSMOS_EXAONE_m1_태우"
FROZEN_OLD = DESIGN / "analysis/out_calib/frozen_beta.json"          # DeepSeek·Haiku (읽기 전용)
FROZEN_EX = DESIGN / "analysis/out_calib/frozen_beta_exaone_k100.json"
ALPHA, ALPHA_J = 1.0, 0.5
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
LAMS = np.round(np.arange(0, 1.0001, 0.01), 2)
B_HUMAN = 1000
SEED = 20261112
rng = np.random.default_rng(SEED)
M = ["exaone", "deepseek", "haiku"]
NAME = {"exaone": "K-EXAONE 2.5", "deepseek": "DeepSeek V4.1 Flash", "haiku": "Haiku 5.5"}
LEVELS = ["c0", "L0", "L1cell"]
K_TARGET = 100
STARTED = datetime.datetime.now().isoformat(timespec="seconds")


class Tee:
    """화면 출력을 run.log에도 그대로 남긴다."""

    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


sys.stdout = Tee(OUT / "run.log")
print(f"[시작] {STARTED}  스크립트 {Path(__file__).name}")


def sha256_file(p, n=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


# ── 문항 축 ──────────────────────────────────────────────
items = pd.read_csv(DESIGN / "item_manifest_v2.csv", encoding="utf-8-sig")
items["codes"] = items["codes"].apply(json.loads)
I, K = len(items), int(items["codes"].str.len().max())
CODES = np.full((I, K), np.nan)
for i, c in enumerate(items["codes"]):
    CODES[i, :len(c)] = c
MASK = ~np.isnan(CODES)
NOPT = MASK.sum(1)
UNIF = MASK / NOPT[:, None]
SPLIT = {s: (items["split_60_20_20"] == s).to_numpy() for s in ["calibration", "validation", "test"]}
SPLIT["cal+val"] = ~SPLIT["test"]
SPLIT["all"] = np.ones(I, bool)
ORDINAL = items["scale_type"].isin(["ordinal_monotone", "binary"]).to_numpy()

# ── 페르소나 축, 사람 정답지 H, 유효 표본 NEFF ─────────────────
cells = pd.read_csv(DESIGN / "cells.csv", encoding="utf-8-sig").query("level == 'L1'").reset_index(drop=True)
PERSONAS = ["c0", "L0", *cells["cell_id"]]
L1 = slice(2, None)
kgss = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet").query("YEAR == 2023")
kgss = kgss.assign(age_band=pd.cut(kgss["AGE"], [18, 30, 40, 50, 60, 200], right=False,
                                   labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str))
SEX = {"남자": 1, "여자": 2}
resp = {"c0": kgss, "L0": kgss} | {c.cell_id: kgss[(kgss["SEX"] == SEX[c.sex]) & (kgss["age_band"] == c.age_band)]
                                   for c in cells.itertuples()}
H = np.zeros((len(PERSONAS), I, K))
NEFF = np.zeros((len(PERSONAS), I), dtype=int)
for j, p in enumerate(PERSONAS):
    for i, v in enumerate(items["var"]):
        d = resp[p][[v, "FINALWT"]].dropna()
        d = d[d[v].isin(CODES[i, MASK[i]])]
        w = d.groupby(v)["FINALWT"].sum().reindex(CODES[i, MASK[i]]).fillna(0).to_numpy()
        H[j, i, :NOPT[i]] = w / w.sum()
        NEFF[j, i] = round(d["FINALWT"].sum() ** 2 / (d["FINALWT"] ** 2).sum())
assert np.allclose(H.sum(-1), 1)

# ── 원응답 읽기 ──────────────────────────────────────────
P_IDX = {p: j for j, p in enumerate(PERSONAS)}
I_IDX = {v: i for i, v in enumerate(items["var"])}
K_IDX = [{c: k for k, c in enumerate(cs)} for cs in items["codes"]]
probes = {r["probe_id"]: r for r in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig"))
          if r["phase"] == "1"}
SOURCES = {"exaone": PKG / "exaone_p1_merged.jsonl",
           "deepseek": ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl",
           "haiku": ROOT / "raw/v2/haiku_p1.jsonl"}


def read_file(path, store, models_seen):
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_type") == "api_error" or r["probe_id"] not in probes:
            continue
        models_seen[(r.get("model_id"), r.get("reasoning_effort") or json.dumps(r.get("params", {}), sort_keys=True))] += 1
        store.setdefault((r["probe_id"], r["sample_idx"]), (bool(r["valid"]), r.get("parsed"), r.get("invalid_type") or ""))


raw_recs, model_ids = {m: {} for m in M}, {m: collections.Counter() for m in M}
for m in M:
    read_file(SOURCES[m], raw_recs[m], model_ids[m])

have = {m: collections.defaultdict(set) for m in M}
for m in M:
    for pid, s in raw_recs[m]:
        have[m][pid].add(s)
k_per = {m: np.array([len(have[m][pid]) for pid in probes]) for m in M}
for m in M:
    assert (k_per[m] == K_TARGET).all(), f"{m}: 프롬프트당 표본 수가 {K_TARGET}이 아님 (min {k_per[m].min()})"
    assert all(have[m][pid] == set(range(K_TARGET)) for pid in probes), f"{m}: 표본 번호가 0~{K_TARGET - 1}이 아님"
print("표본 수: 세 모델 모두 프롬프트", len(probes), "개 × 표본 번호 0~99 (k=100)")

C = np.zeros((len(M), len(PERSONAS), I, K))
NTOT = np.zeros((len(M), len(PERSONAS), I))
NREF = np.zeros((len(M), len(PERSONAS), I))
for a, m in enumerate(M):
    for (pid, s), (valid, parsed, inv) in raw_recs[m].items():
        p = probes[pid]
        j, i = P_IDX[p["cell_id"]], I_IDX[p["var"]]
        NTOT[a, j, i] += 1
        NREF[a, j, i] += inv == "refusal"
        if valid:
            C[a, j, i, K_IDX[i][parsed]] += 1
COUNTS_SHA = {m: sha256_array(C[a]) for a, m in enumerate(M)}
np.savez_compressed(OUT / "counts_k100.npz", C=C, NTOT=NTOT, NREF=NREF, H=H, NEFF=NEFF, CODES=CODES,
                    models=np.array(M), personas=np.array(PERSONAS), vars=items["var"].to_numpy(),
                    split=items["split_60_20_20"].to_numpy())


# ── 공통 함수 (노트북과 같음) ──────────────────────────────
def to_prob(X, beta=1.0, alpha=ALPHA):
    x = (X + alpha * MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def share(X):
    return X / np.maximum(X.sum(-1, keepdims=True), 1)


def tvd(p, h):
    return 0.5 * np.abs(p - h).sum(-1)


def w1(p, h):
    gap = np.nan_to_num(np.diff(CODES, axis=1))
    span = np.nanmax(CODES, 1) - np.nanmin(CODES, 1)
    d = np.abs(np.cumsum(p, -1) - np.cumsum(h, -1))[..., :-1]
    return np.where(ORDINAL, (d * gap).sum(-1) / span, np.nan)


top = lambda x: np.where(MASK, x, -np.inf).argmax(-1)


def mode_hit(p, h):
    return (top(p) == top(h)).astype(float)


METRICS = {"tvd": tvd, "w1": w1, "mode": mode_hit}


def evaluate(P):
    return {n: np.stack([f(P[..., 0, :, :], H[0]), f(P[..., 1, :, :], H[1]), f(P[..., L1, :, :], H[L1]).mean(-2)], -2)
            for n, f in METRICS.items()}


def mean_ci(x, B=5000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    b = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def by_level(x):
    return np.stack([np.nanmean(x[..., 0, :], -1), np.nanmean(x[..., 1, :], -1), np.nanmean(x[..., L1, :], (-1, -2))], -1)


def norm_entropy(p):
    return -(p * np.log(np.where(p > 0, p, 1))).sum(-1) / np.log(NOPT)


pick = lambda g, ib: np.take_along_axis(g, ib[None, ..., None], 0)[0]
R = share(C)

# ── 0. 유효율 ────────────────────────────────────────────
rows = []
for a, m in enumerate(M):
    for lv, sel in {"c0": [0], "L0": [1], "L1": list(range(2, 12))}.items():
        nt, nv, nr = NTOT[a, sel].sum(), C[a, sel].sum(), NREF[a, sel].sum()
        rows.append(dict(model=m, level=lv, n_total=int(nt), n_valid=int(nv), valid_rate=nv / nt, refusal=int(nr)))
validity = pd.DataFrame(rows)
validity.to_csv(OUT / "validity.csv", index=False)

# ── 1. β 그리드 → calibration β* → 고정 ──────────────────────
G = evaluate(to_prob(C[None], BETAS[:, None, None, None, None]))          # {지표: [β, m, 단계, 문항]}
cal = SPLIT["calibration"]
ib_cal = G["tvd"][..., cal].mean(-1).argmin(0)                            # [m, 단계] (이번 실행 재계산)
GJ = evaluate(to_prob(C[None], BETAS[:, None, None, None, None], ALPHA_J))["tvd"]
ib_cal_j = GJ[..., cal].mean(-1).argmin(0)
P1 = to_prob(C, 1.0)
mixT = np.stack([[tvd((1 - l) * P1[a] + l * UNIF, H) for l in LAMS] for a in range(len(M))])   # [m, λ, p, i]
mixL = np.stack([mixT[..., 0, :], mixT[..., 1, :], mixT[..., L1, :].mean(-2)], -2)          # [m, λ, 단계, i]
il_cal = mixL[..., cal].mean(-1).argmin(1)                                                  # [m, 단계]

ex = M.index("exaone")
new_freeze = {
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": "k-exaone_v2.5 (reasoning_effort=low)", "k_per_prompt": K_TARGET,
    "source": str(SOURCES["exaone"].relative_to(ROOT)),
    "counts_sha256": COUNTS_SHA["exaone"], "primary_alpha": "laplace", "alphas": {"laplace": ALPHA, "jeffreys": ALPHA_J},
    "beta_grid": [float(BETAS[0]), float(BETAS[-1]), 0.005], "criterion": "calibration(148) mean TVD",
    "beta": {**{f"laplace|exaone|{lv}": float(BETAS[ib_cal[ex, l]]) for l, lv in enumerate(LEVELS)},
             **{f"jeffreys|exaone|{lv}": float(BETAS[ib_cal_j[ex, l]]) for l, lv in enumerate(LEVELS)}},
    "lam": {f"exaone|{lv}": float(LAMS[il_cal[ex, l]]) for l, lv in enumerate(LEVELS)},
    "script": Path(__file__).name,
}
if FROZEN_EX.exists():
    frozen_ex = json.loads(FROZEN_EX.read_text(encoding="utf-8"))
    assert frozen_ex["counts_sha256"] == COUNTS_SHA["exaone"], "EXAONE 개수가 고정 당시와 다르다. 고정 파일을 덮어쓰지 말고 원인을 확인할 것"
    freeze_status = f"기존 고정값 사용 ({frozen_ex['frozen_at']})"
else:
    frozen_ex = new_freeze
    FROZEN_EX.write_text(json.dumps(frozen_ex, ensure_ascii=False, indent=1), encoding="utf-8")
    freeze_status = f"이번 실행에서 고정 ({frozen_ex['frozen_at']})"
print("EXAONE β 고정:", freeze_status, "→", FROZEN_EX.name)
frozen_old = json.loads(FROZEN_OLD.read_text(encoding="utf-8"))

BI = {round(float(b), 3): n for n, b in enumerate(BETAS)}
LI = {round(float(x), 2): n for n, x in enumerate(LAMS)}
ib_frozen = np.zeros_like(ib_cal)
il_frozen = np.zeros_like(il_cal)
check = []
for a, m in enumerate(M):
    src = frozen_ex if m == "exaone" else frozen_old
    for l, lv in enumerate(LEVELS):
        b, lam = src["beta"][f"laplace|{m}|{lv}"], src["lam"][f"{m}|{lv}"]
        ib_frozen[a, l], il_frozen[a, l] = BI[round(b, 3)], LI[round(lam, 2)]
        check.append(dict(model=m, level=lv, beta_frozen=b, beta_recomputed=float(BETAS[ib_cal[a, l]]),
                          lam_frozen=lam, lam_recomputed=float(LAMS[il_cal[a, l]]),
                          beta_jeffreys_recomputed=float(BETAS[ib_cal_j[a, l]]),
                          frozen_file=FROZEN_EX.name if m == "exaone" else FROZEN_OLD.name))
beta_check = pd.DataFrame(check)
beta_check["match"] = (beta_check.beta_frozen == beta_check.beta_recomputed) & (beta_check.lam_frozen == beta_check.lam_recomputed)
beta_check.to_csv(OUT / "beta_frozen_check.csv", index=False)

i_one, i_34 = np.argmin(abs(BETAS - 1)), np.argmin(abs(BETAS - 0.75))
SETTINGS = {"β=1": np.full_like(ib_cal, i_one), "β=0.75": np.full_like(ib_cal, i_34), "β*": ib_frozen}

# ── 2. 지표표 (test 공개) ──────────────────────────────────
TEST_OPENED = True
unif = evaluate(np.broadcast_to(UNIF, (len(PERSONAS), I, K)))
rows = []
for split in ["calibration", "validation", "test", "cal+val", "all"]:
    sel = SPLIT[split]
    for met, g in G.items():
        for st, ib in SETTINGS.items():
            x = pick(g, ib)
            for a, m in enumerate(M):
                for l, lv in enumerate(LEVELS):
                    mu, lo, hi = mean_ci(x[a, l, sel])
                    rows.append(dict(split=split, metric=met, model=m, level=lv, setting=st,
                                     beta=float(BETAS[ib[a, l]]), mean=mu, lo=lo, hi=hi,
                                     beat_unif=float((x[a, l, sel] < unif[met][l, sel]).mean()) if met != "mode" else np.nan))
        if met != "mode":
            for l, lv in enumerate(LEVELS):
                rows.append(dict(split=split, metric=met, model="uniform", level=lv, setting="",
                                 mean=float(np.nanmean(unif[met][l, sel]))))
    for a, m in enumerate(M):
        for l, lv in enumerate(LEVELS):
            mu, lo, hi = mean_ci(mixL[a, il_frozen[a, l], l, sel])
            rows.append(dict(split=split, metric="tvd", model=m, level=lv, setting="균등혼합 λ*",
                             beta=float(LAMS[il_frozen[a, l]]), mean=mu, lo=lo, hi=hi))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "metrics_by_setting.csv", index=False)

# ── 3. 모델 간 쌍대 차이 (같은 문항) + β* vs 균등혼합 ────────────
rows = []
for split, st in [("test", "β*"), ("test", "β=1"), ("validation", "β*"), ("cal+val", "β=1"), ("all", "β=1")]:
    for met, g in G.items():
        x = pick(g, SETTINGS[st])
        for b_, a_ in [("exaone", "deepseek"), ("exaone", "haiku"), ("deepseek", "haiku")]:
            for l, lv in enumerate(LEVELS):
                d = x[M.index(b_), l] - x[M.index(a_), l]
                mu, lo, hi = mean_ci(d[SPLIT[split]])
                rows.append(dict(split=split, setting=st, metric=met, contrast=f"{b_} − {a_}", level=lv,
                                 diff=mu, lo=lo, hi=hi))
xb = pick(G["tvd"], ib_frozen)
for a, m in enumerate(M):
    for l, lv in enumerate(LEVELS):
        d = xb[a, l] - mixL[a, il_frozen[a, l], l]
        mu, lo, hi = mean_ci(d[SPLIT["test"]])
        rows.append(dict(split="test", setting="β* − 균등혼합 λ*", metric="tvd", contrast=m, level=lv, diff=mu, lo=lo, hi=hi))
paired = pd.DataFrame(rows)
paired.to_csv(OUT / "paired_diffs.csv", index=False)

# ── 4. β와 무관한 지표: 범위(test / cal+val / all)마다 ─────────────
Hb = np.zeros((B_HUMAN, *H.shape))
for j in range(len(PERSONAS)):
    for i in range(I):
        Hb[:, j, i, :NOPT[i]] = rng.multinomial(NEFF[j, i], H[j, i, :NOPT[i]], size=B_HUMAN) / NEFF[j, i]
hlo, hhi = np.percentile(Hb, [2.5, 97.5], axis=0)
top_h = top(H)
lo_top = np.take_along_axis(hlo, top_h[..., None], -1)[..., 0]
GROUP = MASK & (hhi >= lo_top[..., None])


def trm(P):
    mm = np.where(MASK, P, -np.inf)
    is_top = MASK & (mm == mm.max(-1, keepdims=True))
    return (is_top & GROUP).sum(-1) / is_top.sum(-1)


up = np.triu(np.ones((10, 10), bool), 1)
sex, age = cells["sex"].to_numpy(), cells["age_band"].to_numpy()
KIND = {"전체": np.ones((10, 10), bool), "성별만 다름": age[:, None] == age[None, :], "연령만 다름": sex[:, None] == sex[None, :]}
score = lambda P: (P * np.nan_to_num(CODES)).sum(-1)
cd = lambda x: x[..., :, None, :] - x[..., None, :, :]
dh = cd(score(H[L1]))
dlo, dhi = np.percentile(cd(score(Hb[:, L1])), [2.5, 97.5], axis=0)
SIG = (dlo > 0) | (dhi < 0)
trm_hb = np.mean([trm(Hb[b]) for b in range(0, B_HUMAN, 10)], 0)
T1 = G["tvd"][i_one]                                                     # [m, 단계, i]
sens = items["sensitivity_level"].to_numpy(float)


def auc_with_ci(dm, valid, nb=2000):
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    hit, n = (s * valid).sum((0, 1)), valid.sum((0, 1)).astype(float)
    idx = rng.choice(np.where(n > 0)[0], (nb, int((n > 0).sum())))
    b = hit[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    tie = float(((dm == 0) & valid).sum() / valid.sum())
    return float(hit.sum() / n.sum()), *np.percentile(b, [2.5, 97.5]), int(n.sum()), tie


def pair_auc(dm, dhh, valid):
    s = np.where(np.sign(dm) == np.sign(dhh), 1.0, np.where(dm == 0, 0.5, 0.0))
    v = valid & (dhh != 0)
    w = np.abs(dhh) * v
    return float((s * v).sum() / v.sum()), float((s * w).sum() / w.sum())


def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


def secondary(EV):
    out = {}
    rows = []
    for a, m in enumerate(M):
        for name, (l0, l1) in {"c0 → L0": (0, 1), "L0 → L1cell": (1, 2)}.items():
            d = (T1[a, l1] - T1[a, l0])[EV]
            mu, lo, hi = mean_ci(d)
            rows.append(dict(model=m, step=name, tvd_change=mu, lo=lo, hi=hi, improved_share=float((d < 0).mean())))
        rows[-2]["mean_shift_tvd"] = float(tvd(R[a, 0], R[a, 1])[EV].mean())
        rows[-2]["mode_changed"] = float((top(R[a, 0]) != top(R[a, 1]))[EV].mean())
    out["persona_steps"] = pd.DataFrame(rows)

    rows = []
    for a, m in enumerate(M):
        for l, lv in enumerate(LEVELS):
            rows.append(dict(model=m, level=lv, entropy=float(by_level(norm_entropy(R[a])[:, EV])[l]),
                             top_share=float(by_level(R[a].max(-1)[:, EV])[l]),
                             top_ge90=float(by_level((R[a].max(-1) >= 0.9).astype(float)[:, EV])[l]),
                             invalid=float(by_level((1 - C[a].sum(-1) / np.maximum(NTOT[a], 1))[:, EV])[l])))
    for l, lv in enumerate(LEVELS):
        rows.append(dict(model="human", level=lv, entropy=float(by_level(norm_entropy(H)[:, EV])[l]),
                         top_share=float(by_level(H.max(-1)[:, EV])[l]),
                         top_ge90=float(by_level((H.max(-1) >= 0.9).astype(float)[:, EV])[l])))
    out["collapse_ladder"] = pd.DataFrame(rows)

    rows = []
    lv3 = lambda x: by_level(x[:, EV])
    for a, m in enumerate(M):
        for nm, x in [("엄격(1위 일치)", (top(R[a]) == top_h).astype(float)), ("RADIUS", trm(R[a]))]:
            rows.append(dict(기준=f"{m} · {nm}", **dict(zip(LEVELS, lv3(x)))))
    rows.append(dict(기준="우연 · 엄격", **dict(zip(LEVELS, lv3(np.broadcast_to(1 / NOPT, top_h.shape))))))
    rows.append(dict(기준="우연 · RADIUS", **dict(zip(LEVELS, lv3(GROUP.sum(-1) / NOPT)))))
    rows.append(dict(기준="모집단 사람 분포 · RADIUS", **dict(zip(LEVELS, lv3(trm(np.broadcast_to(H[0], H.shape)))))))
    rows.append(dict(기준="사람 재표집 · RADIUS (상한)", **dict(zip(LEVELS, lv3(trm_hb)))))
    out["mode_radius"] = pd.DataFrame(rows)

    rows = []
    for kind, km in KIND.items():
        base = up[..., None] & km[..., None] & (ORDINAL & EV)[None, None, :] & (dh != 0)
        for a, m in enumerate(M):
            dm = cd(score(R[a, L1]))
            for which, v in [("유의쌍만(주)", base & SIG), ("전체쌍", base)]:
                auc, lo, hi, n, tie = auc_with_ci(dm, v)
                rows.append(dict(쌍=kind, 범위=which, model=m, AUC=auc, lo=lo, hi=hi, 쌍수=n, 모델동점=tie))
    out["pairs_radius"] = pd.DataFrame(rows)

    def option_pairs(p, h):
        d = lambda x: x[..., :, None] - x[..., None, :]
        valid = np.triu(np.ones((K, K), bool), 1) & MASK[:, :, None] & MASK[:, None, :] & EV[:, None, None]
        return d(p), d(h), np.broadcast_to(valid, d(h).shape)

    rows = []
    for a, m in enumerate(M):
        for l, sel in enumerate([[0], [1], list(range(2, 12))]):
            rows.append(dict(model=m, level=LEVELS[l], **dict(zip(["AUC", "가중AUC"], pair_auc(*option_pairs(R[a, sel], H[sel]))))))
    rows.append(dict(model="기준선: 모집단 사람 순서", level="L1cell",
                     **dict(zip(["AUC", "가중AUC"], pair_auc(*option_pairs(np.broadcast_to(H[0], H[L1].shape), H[L1]))))))
    out["bpr_option_pairs"] = pd.DataFrame(rows)

    hm = score(H[L1])
    OE = ORDINAL & EV
    spread = {}
    for a, m in enumerate(M):
        ms = score(R[a, L1])
        spread[m] = {"셀 간 평균 TVD": float(tvd(R[a, L1][:, None], R[a, L1][None, :])[up][:, EV].mean()),
                     "SD 비 모델/사람 (중앙값)": float(np.median((ms.std(0) / hm.std(0))[OE])),
                     "셀 평균점수 상관 (중앙값)": float(np.nanmedian([np.corrcoef(ms[:, i], hm[:, i])[0, 1] if ms[:, i].std() > 0 else np.nan
                                                       for i in np.where(OE)[0]])),
                     "L0 대비 셀 이동 TVD": float(tvd(R[a, L1], R[a, 1])[:, EV].mean())}
    spread["human"] = {"셀 간 평균 TVD": float(tvd(H[L1][:, None], H[L1][None, :])[up][:, EV].mean()),
                       "SD 비 모델/사람 (중앙값)": 1.0, "셀 평균점수 상관 (중앙값)": 1.0,
                       "L0 대비 셀 이동 TVD": float(tvd(H[L1], H[1])[:, EV].mean())}
    out["cell_spread"] = pd.DataFrame(spread).reset_index(names="지표")

    rows = []
    for a, m in enumerate(M):
        for lv, sel in {"c0": [0], "L0": [1], "L1": list(range(2, 12))}.items():
            exc = (R[a, sel].max(-1) - H[sel].max(-1)).mean(0)
            x, y = sens[EV], exc[EV]
            bs = [spearman(x[ix], y[ix]) for ix in rng.integers(0, len(x), (1000, len(x)))]
            rows.append(dict(model=m, level=lv, rho_sens_excess=spearman(x, y), lo=float(np.percentile(bs, 2.5)),
                             hi=float(np.percentile(bs, 97.5)), mean_excess=float(y.mean())))
    out["sensitivity_collapse"] = pd.DataFrame(rows)
    return out


SEC = {}
for scope in ["test", "cal+val", "all"]:
    for name, t in secondary(SPLIT[scope]).items():
        SEC.setdefault(name, []).append(t.assign(scope=scope))
SEC = {name: pd.concat(ts, ignore_index=True) for name, ts in SEC.items()}
for name, t in SEC.items():
    t.to_csv(OUT / f"{name}.csv", index=False)


# ── 요약 · 기록 ───────────────────────────────────────────
def mrow(split, met, st, m, lv):
    r = metrics[(metrics.split == split) & (metrics.metric == met) & (metrics.setting == ("" if m == "uniform" else st)) &
                (metrics.model == m) & (metrics.level == lv)].iloc[0]
    return [round(r["mean"], 4), round(r.lo, 4) if pd.notna(r.lo) else None, round(r.hi, 4) if pd.notna(r.hi) else None]


HEAD_SETS = [("test", "β*"), ("test", "β=1"), ("validation", "β*"), ("cal+val", "β=1")]
summary = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "k_per_prompt": K_TARGET, "items": {s: int(v.sum()) for s, v in SPLIT.items()}, "test_opened": TEST_OPENED,
    "model_settings": {m: [list(k) + [v] for k, v in model_ids[m].items()] for m in M},
    "freeze": {"status": freeze_status, "exaone_file": str(FROZEN_EX.relative_to(ROOT)), "exaone": frozen_ex,
               "deepseek_haiku_file": str(FROZEN_OLD.relative_to(ROOT))},
    "beta_check": beta_check.to_dict("records"),
    "validity": validity.round(5).to_dict("records"),
    "headline": {f"{split}|{met}|{st}|{lv}": {m: mrow(split, met, st, m, lv) for m in [*M, *(["uniform"] if met != "mode" and st == "β=1" else [])]}
                 for split, st in HEAD_SETS for met in ["tvd", "w1", "mode"] for lv in LEVELS},
    "paired": paired.round(4).to_dict("records"),
    **{name: t.round(4).to_dict("records") for name, t in SEC.items()},
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


inputs = {**{f"raw:{m}": p for m, p in SOURCES.items()},
          "item_manifest": DESIGN / "item_manifest_v2.csv", "cells": DESIGN / "cells.csv",
          "prompts": DESIGN / "prompts_v2.csv", "kgss": ROOT / "inventory/kgss_clean.parquet",
          "frozen_deepseek_haiku": FROZEN_OLD, "frozen_exaone": FROZEN_EX}
run = {
    "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256_file(__file__),
    "command": " ".join([Path(sys.executable).name, *sys.argv]),
    "started": STARTED, "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "--short", "HEAD") or None, "git_branch": git("branch", "--show-current") or None,
    "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "include_test": TEST_OPENED,
    "inputs": {k: {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for k, p in inputs.items()},
    "counts_sha256": COUNTS_SHA,
    "params": {"alpha": ALPHA, "alpha_sensitivity": ALPHA_J, "beta_grid": [float(BETAS[0]), float(BETAS[-1]), 0.005],
               "lambda_grid": [0.0, 1.0, 0.01], "B_human": B_HUMAN, "seed": SEED, "k": K_TARGET,
               "bootstrap_mean_ci": 5000, "bootstrap_pairs": 2000, "bootstrap_spearman": 1000},
    "freeze_status": freeze_status,
    "outputs": sorted(p.name for p in OUT.iterdir() if p.name != "run.json"),
    "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
}
(OUT / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")

pd.set_option("display.width", 220)
print(validity.round(4).to_string(index=False))
print("\n[β 고정값 점검]\n", beta_check.to_string(index=False))
for split, st in HEAD_SETS:
    for met in ["tvd", "w1", "mode"]:
        t = metrics[(metrics.split == split) & (metrics.metric == met) & (metrics.setting.isin([st, ""]))]
        print(f"\n[{split} · {met} · {st}]\n", t.pivot_table(index="level", columns="model", values="mean", sort=False).reindex(LEVELS).round(3))
print("\n[균등혼합 λ*, test]\n", metrics[(metrics.split == "test") & (metrics.setting == "균등혼합 λ*")]
      .pivot_table(index="level", columns="model", values="mean", sort=False).reindex(LEVELS).round(3))
print("\n", paired.round(3).to_string(index=False))
for name, t in SEC.items():
    print(f"\n[{name}]\n", t.round(3).to_string(index=False))
print(f"\n[끝] {run['finished']}  git {run['git_commit']} dirty={run['git_dirty']}  → {OUT.relative_to(ROOT)}")
