"""K-EXAONE 중간 결과를 설계 v2 본 분석 프레임으로 DeepSeek·Haiku와 비교한다 (API 호출 없음).

프레임 (v2_experiment.ipynb · radius_ties.py와 같은 규칙)
- 정답지: KGSS 2023, FINALWT 가중. 페르소나 12개(c0, L0, L1 10셀).
- 보정: 라플라스 α = 1, p ∝ (c + 1)^β. β 그리드 0.05~1.5(0.005 간격), calibration 148문항 평균 TVD 최소로 β* 선택.
- 지표: TVD, 정규화 W1(서열·이진), 최빈값 정확도(엄격 · RADIUS 동점 규칙), BPR식 쌍별 AUC, 셀 간 쌍별 순위.
- 단계: c0, L0, L1cell(10셀 각각의 지표 평균).

이번 분석에서 다른 점
- 표본 수 맞춤: EXAONE이 아직 다 모이지 않아, 프롬프트마다 세 모델 모두 같은 표본 번호(EXAONE이 가진 번호)만 쓴다.
  가용한 표본은 모두 쓴다(프롬프트마다 k가 조금씩 다르다).
- test 49문항은 열지 않는다. EXAONE의 test는 k=100이 모인 뒤 한 번만 평가한다(DESIGN.md 6장).
  β*는 calibration에서 고르고 validation에서 확인한다. β와 무관한 지표는 calibration+validation 197문항으로 본다.
- frozen_beta.json은 읽지도 쓰지도 않는다(표본 수가 달라 β*가 달라진다).

실행 (KOSMOS 루트):
  .venv-nb/bin/python research/design_v2_20261112/analysis/exaone_interim_v2.py
산출: analysis/out_exaone_interim/*.csv, summary.json
"""
from __future__ import annotations

import collections
import csv
import datetime
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "inventory/kgss_clean.parquet").exists())
DESIGN = ROOT / "research/design_v2_20261112"
OUT = DESIGN / "analysis/out_exaone_interim"
OUT.mkdir(parents=True, exist_ok=True)
PKG = DESIGN / "team_packages/KOSMOS_EXAONE_m1_태우"
ALPHA = 1.0
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
LAMS = np.round(np.arange(0, 1.0001, 0.01), 2)
B_HUMAN = 1000
rng = np.random.default_rng(20261112)
M = ["exaone", "deepseek", "haiku"]
NAME = {"exaone": "K-EXAONE 2.5", "deepseek": "DeepSeek V4.1 Flash", "haiku": "Haiku 5.5"}
LEVELS = ["c0", "L0", "L1cell"]

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
ORDINAL = items["scale_type"].isin(["ordinal_monotone", "binary"]).to_numpy()
EV = SPLIT["cal+val"]

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


def read_lines(lines, store, models_seen):
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:          # 실행 중 파일의 쓰는 중인 마지막 줄
            continue
        if r.get("invalid_type") == "api_error" or r["probe_id"] not in probes:
            continue
        models_seen[(r.get("model_id"), r.get("reasoning_effort") or json.dumps(r.get("params", {}), sort_keys=True))] += 1
        store.setdefault((r["probe_id"], r["sample_idx"]), (bool(r["valid"]), r.get("parsed"), r.get("invalid_type") or ""))


raw_recs, sources, model_ids = {m: {} for m in M}, {m: [] for m in M}, {m: collections.Counter() for m in M}
for z in sorted(PKG.glob("KOSMOS_EXAONE_결과_m*_*.zip")):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.endswith(".jsonl"):
                read_lines(io.TextIOWrapper(zf.open(n), encoding="utf-8"), raw_recs["exaone"], model_ids["exaone"])
                sources["exaone"].append(f"{z.name}:{n}")
for f in sorted(PKG.glob("exaone_m*_p1.jsonl")):
    read_lines(open(f, encoding="utf-8"), raw_recs["exaone"], model_ids["exaone"])
    sources["exaone"].append(f.name)
for m, f in [("deepseek", ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl"), ("haiku", ROOT / "raw/v2/haiku_p1.jsonl")]:
    read_lines(open(f, encoding="utf-8"), raw_recs[m], model_ids[m])
    sources[m].append(str(f.relative_to(ROOT)))

# 프롬프트마다 세 모델 공통 표본 번호
have = {m: collections.defaultdict(set) for m in M}
for m in M:
    for pid, s in raw_recs[m]:
        have[m][pid].add(s)
S = {pid: have["exaone"][pid] & have["deepseek"][pid] & have["haiku"][pid] for pid in probes}
k_per = np.array([len(S[pid]) for pid in probes])

C = np.zeros((len(M), len(PERSONAS), I, K))
NTOT = np.zeros((len(M), len(PERSONAS), I))
NREF = np.zeros((len(M), len(PERSONAS), I))
for a, m in enumerate(M):
    for (pid, s), (valid, parsed, inv) in raw_recs[m].items():
        if s not in S[pid]:
            continue
        p = probes[pid]
        j, i = P_IDX[p["cell_id"]], I_IDX[p["var"]]
        NTOT[a, j, i] += 1
        NREF[a, j, i] += inv == "refusal"
        if valid:
            C[a, j, i, K_IDX[i][parsed]] += 1
# k=100 전체 (DeepSeek·Haiku만, 표본 수 영향 확인용)
C100 = np.zeros((2, len(PERSONAS), I, K))
for a, m in enumerate(["deepseek", "haiku"]):
    for (pid, s), (valid, parsed, inv) in raw_recs[m].items():
        if valid:
            p = probes[pid]
            i = I_IDX[p["var"]]
            C100[a, P_IDX[p["cell_id"]], i, K_IDX[i][parsed]] += 1


# ── 공통 함수 (노트북과 같음) ──────────────────────────────
def to_prob(X, beta=1.0):
    x = (X + ALPHA * MASK) ** beta * MASK
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
summary = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"), "sources": sources,
           "model_settings": {m: [list(k) + [v] for k, v in model_ids[m].items()] for m in M},
           "k_per_prompt": dict(min=int(k_per.min()), median=float(np.median(k_per)), mean=float(k_per.mean()),
                                max=int(k_per.max())),
           "items": {s: int(v.sum()) for s, v in SPLIT.items()}, "test_opened": False}

# ── 0. 유효율 ────────────────────────────────────────────
rows = []
for a, m in enumerate(M):
    for lv, sel in {"c0": [0], "L0": [1], "L1": list(range(2, 12))}.items():
        nt, nv, nr = NTOT[a, sel].sum(), C[a, sel].sum(), NREF[a, sel].sum()
        rows.append(dict(model=m, level=lv, n_total=int(nt), n_valid=int(nv), valid_rate=nv / nt, refusal=int(nr)))
validity = pd.DataFrame(rows)
validity.to_csv(OUT / "validity.csv", index=False)

# ── 1. β 그리드 → β* (calibration) → 지표표 ─────────────────
G = evaluate(to_prob(C[None], BETAS[:, None, None, None, None]))          # {지표: [β, m, 단계, 문항]}
ib_star = G["tvd"][..., SPLIT["calibration"]].mean(-1).argmin(0)        # [m, 단계]
i_one, i_34 = np.argmin(abs(BETAS - 1)), np.argmin(abs(BETAS - 0.75))
SETTINGS = {"β=1": np.full_like(ib_star, i_one), "β=0.75": np.full_like(ib_star, i_34), "β*": ib_star}
beta_star = pd.DataFrame(BETAS[ib_star], index=M, columns=LEVELS)
beta_star.to_csv(OUT / "beta_star.csv")

# 균등 혼합 기준선 λ (calibration에서 고름, β=1 분포 기준)
P1 = to_prob(C, 1.0)
mixT = np.stack([[tvd((1 - l) * P1[a] + l * UNIF, H) for l in LAMS] for a in range(len(M))])   # [m, λ, p, i]
mixL = np.stack([mixT[..., 0, :], mixT[..., 1, :], mixT[..., L1, :].mean(-2)], -2)          # [m, λ, 단계, i]
il_star = mixL[..., SPLIT["calibration"]].mean(-1).argmin(1)                                # [m, 단계]

unif = evaluate(np.broadcast_to(UNIF, (len(PERSONAS), I, K)))
rows = []
for split in ["calibration", "validation", "cal+val"]:
    sel = SPLIT[split]
    for met, g in G.items():
        for st, ib in SETTINGS.items():
            x = pick(g, ib)
            for a, m in enumerate(M):
                for l, lv in enumerate(LEVELS):
                    mu, lo, hi = mean_ci(x[a, l, sel])
                    rows.append(dict(split=split, metric=met, model=m, level=lv, setting=st,
                                     beta=float(BETAS[ib[a, l]]), mean=mu, lo=lo, hi=hi))
        if met != "mode":
            for l, lv in enumerate(LEVELS):
                rows.append(dict(split=split, metric=met, model="uniform", level=lv, setting="",
                                 mean=float(np.nanmean(unif[met][l, sel]))))
    for a, m in enumerate(M):
        for l, lv in enumerate(LEVELS):
            mu, lo, hi = mean_ci(mixL[a, il_star[a, l], l, sel])
            rows.append(dict(split=split, metric="tvd", model=m, level=lv, setting="균등혼합 λ*",
                             beta=float(LAMS[il_star[a, l]]), mean=mu, lo=lo, hi=hi))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "metrics_by_setting.csv", index=False)

# 표본 수 영향: DeepSeek·Haiku k=100 vs 맞춘 k (β=1 TVD, cal+val)
G100 = evaluate(to_prob(C100, 1.0))
kcheck = pd.DataFrame([dict(model=m, level=lv, tvd_matched_k=float(G["tvd"][i_one, M.index(m), l, EV].mean()),
                            tvd_k100=float(G100["tvd"][a, l, EV].mean()))
                       for a, m in enumerate(["deepseek", "haiku"]) for l, lv in enumerate(LEVELS)])
kcheck.to_csv(OUT / "k_sensitivity.csv", index=False)

# ── 2. 모델 간 쌍대 차이 (같은 문항) ──────────────────────────
rows = []
for split, st in [("validation", "β*"), ("cal+val", "β=1")]:
    for met, g in G.items():
        x = pick(g, SETTINGS[st])
        for b_, a_ in [("exaone", "deepseek"), ("exaone", "haiku"), ("deepseek", "haiku")]:
            for l, lv in enumerate(LEVELS):
                d = x[M.index(b_), l] - x[M.index(a_), l]
                mu, lo, hi = mean_ci(d[SPLIT[split]])
                rows.append(dict(split=split, setting=st, metric=met, contrast=f"{b_} − {a_}", level=lv,
                                 diff=mu, lo=lo, hi=hi))
paired = pd.DataFrame(rows)
paired.to_csv(OUT / "paired_diffs.csv", index=False)

# ── 3. 페르소나 단계 효과 (β=1 TVD, cal+val) ──────────────────
rows = []
T1 = G["tvd"][i_one]                                                     # [m, 단계, i]
for a, m in enumerate(M):
    for name, (l0, l1) in {"c0 → L0": (0, 1), "L0 → L1cell": (1, 2)}.items():
        d = (T1[a, l1] - T1[a, l0])[EV]
        mu, lo, hi = mean_ci(d)
        rows.append(dict(model=m, step=name, tvd_change=mu, lo=lo, hi=hi, improved_share=float((d < 0).mean())))
    rows[-2]["mean_shift_tvd"] = float(tvd(R[a, 0], R[a, 1])[EV].mean())
    rows[-2]["mode_changed"] = float((top(R[a, 0]) != top(R[a, 1]))[EV].mean())
line_effect = pd.DataFrame(rows)
line_effect.to_csv(OUT / "persona_steps.csv", index=False)

ladder_rows = []
for a, m in enumerate(M):
    for l, lv in enumerate(LEVELS):
        ladder_rows.append(dict(model=m, level=lv,
                                entropy=float(by_level(norm_entropy(R[a])[:, EV])[l]),
                                top_share=float(by_level(R[a].max(-1)[:, EV])[l]),
                                top_ge90=float(by_level((R[a].max(-1) >= 0.9).astype(float)[:, EV])[l]),
                                invalid=float(by_level((1 - C[a].sum(-1) / np.maximum(NTOT[a], 1))[:, EV])[l])))
for l, lv in enumerate(LEVELS):
    ladder_rows.append(dict(model="human", level=lv, entropy=float(by_level(norm_entropy(H)[:, EV])[l]),
                            top_share=float(by_level(H.max(-1)[:, EV])[l]),
                            top_ge90=float(by_level((H.max(-1) >= 0.9).astype(float)[:, EV])[l])))
ladder = pd.DataFrame(ladder_rows)
ladder.to_csv(OUT / "collapse_ladder.csv", index=False)

# ── 4. RADIUS 동점 규칙: 최빈값, 셀 간 쌍 ───────────────────────
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


rows = []
lv3 = lambda x: by_level(x[:, EV])
for a, m in enumerate(M):
    for nm, x in [("엄격(1위 일치)", (top(R[a]) == top_h).astype(float)), ("RADIUS", trm(R[a]))]:
        rows.append(dict(기준=f"{m} · {nm}", **dict(zip(LEVELS, lv3(x)))))
rows.append(dict(기준="우연 · 엄격", **dict(zip(LEVELS, lv3(np.broadcast_to(1 / NOPT, top_h.shape))))))
rows.append(dict(기준="우연 · RADIUS", **dict(zip(LEVELS, lv3(GROUP.sum(-1) / NOPT)))))
rows.append(dict(기준="모집단 사람 분포 · RADIUS", **dict(zip(LEVELS, lv3(trm(np.broadcast_to(H[0], H.shape)))))))
rows.append(dict(기준="사람 재표집 · RADIUS (상한)", **dict(zip(LEVELS, lv3(np.mean([trm(Hb[b]) for b in range(0, B_HUMAN, 10)], 0))))))
mode_radius = pd.DataFrame(rows)
mode_radius.to_csv(OUT / "mode_radius.csv", index=False)

up = np.triu(np.ones((10, 10), bool), 1)
sex, age = cells["sex"].to_numpy(), cells["age_band"].to_numpy()
KIND = {"전체": np.ones((10, 10), bool), "성별만 다름": age[:, None] == age[None, :], "연령만 다름": sex[:, None] == sex[None, :]}
score = lambda P: (P * np.nan_to_num(CODES)).sum(-1)
cd = lambda x: x[..., :, None, :] - x[..., None, :, :]
dh = cd(score(H[L1]))
dlo, dhi = np.percentile(cd(score(Hb[:, L1])), [2.5, 97.5], axis=0)
SIG = (dlo > 0) | (dhi < 0)


def auc_with_ci(dm, valid, nb=2000):
    s = np.where(np.sign(dm) == np.sign(dh), 1.0, np.where(dm == 0, 0.5, 0.0))
    hit, n = (s * valid).sum((0, 1)), valid.sum((0, 1)).astype(float)
    idx = rng.choice(np.where(n > 0)[0], (nb, int((n > 0).sum())))
    b = hit[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    tie = float(((dm == 0) & valid).sum() / valid.sum())
    return float(hit.sum() / n.sum()), *np.percentile(b, [2.5, 97.5]), int(n.sum()), tie


rows = []
for kind, km in KIND.items():
    base = up[..., None] & km[..., None] & (ORDINAL & EV)[None, None, :] & (dh != 0)
    for a, m in enumerate(M):
        dm = cd(score(R[a, L1]))
        for which, v in [("유의쌍만(주)", base & SIG), ("전체쌍", base)]:
            auc, lo, hi, n, tie = auc_with_ci(dm, v)
            rows.append(dict(쌍=kind, 범위=which, model=m, AUC=auc, lo=lo, hi=hi, 쌍수=n, 모델동점=tie))
pairs_radius = pd.DataFrame(rows)
pairs_radius.to_csv(OUT / "pairs_radius.csv", index=False)

# ── 5. BPR식 선지 간 AUC (셀 안 보기 순서) ─────────────────────
def pair_auc(dm, dhh, valid):
    s = np.where(np.sign(dm) == np.sign(dhh), 1.0, np.where(dm == 0, 0.5, 0.0))
    v = valid & (dhh != 0)
    w = np.abs(dhh) * v
    return float((s * v).sum() / v.sum()), float((s * w).sum() / w.sum())


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
bpr = pd.DataFrame(rows)
bpr.to_csv(OUT / "bpr_option_pairs.csv", index=False)

# ── 6. 셀 간 퍼짐 ─────────────────────────────────────────
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
spread = pd.DataFrame(spread)
spread.to_csv(OUT / "cell_spread.csv")

# ── 7. 민감도와 쏠림 (스피어만, L0) ───────────────────────────
def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))


sens = items["sensitivity_level"].to_numpy(float)
rows = []
for a, m in enumerate(M):
    for lv, sel in {"c0": [0], "L0": [1], "L1": list(range(2, 12))}.items():
        exc = (R[a, sel].max(-1) - H[sel].max(-1)).mean(0)
        x, y = sens[EV], exc[EV]
        bs = [spearman(x[ix], y[ix]) for ix in rng.integers(0, len(x), (1000, len(x)))]
        rows.append(dict(model=m, level=lv, rho_sens_excess=spearman(x, y), lo=float(np.percentile(bs, 2.5)),
                         hi=float(np.percentile(bs, 97.5)), mean_excess=float(y.mean())))
sens_tbl = pd.DataFrame(rows)
sens_tbl.to_csv(OUT / "sensitivity_collapse.csv", index=False)

# ── 요약 ─────────────────────────────────────────────────
def mrow(split, met, st, m, lv):
    r = metrics[(metrics.split == split) & (metrics.metric == met) & (metrics.setting == ("" if m == "uniform" else st)) &
                (metrics.model == m) & (metrics.level == lv)].iloc[0]
    return [round(r["mean"], 4), round(r.lo, 4) if pd.notna(r.lo) else None, round(r.hi, 4) if pd.notna(r.hi) else None]


summary.update(
    validity=validity.round(5).to_dict("records"),
    beta_star=beta_star.to_dict(),
    headline={f"{split}|{met}|{st}|{lv}": {m: mrow(split, met, st, m, lv) for m in [*M, *(["uniform"] if met != "mode" and st == "β=1" else [])]}
              for split, st in [("validation", "β*"), ("validation", "β=1"), ("cal+val", "β=1")]
              for met in ["tvd", "w1", "mode"] for lv in LEVELS},
    paired=paired.round(4).to_dict("records"),
    persona_steps=line_effect.round(4).to_dict("records"),
    ladder=ladder.round(4).to_dict("records"),
    mode_radius=mode_radius.round(4).to_dict("records"),
    pairs_radius=pairs_radius.round(4).to_dict("records"),
    bpr=bpr.round(4).to_dict("records"),
    spread=spread.round(4).to_dict(),
    sensitivity=sens_tbl.round(4).to_dict("records"),
    k_sensitivity=kcheck.round(4).to_dict("records"),
)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

pd.set_option("display.width", 220)
print("k/프롬프트:", summary["k_per_prompt"], "| 출처:", sources)
print(validity.round(4).to_string(index=False))
print("\nβ*\n", beta_star)
for split, st in [("validation", "β*"), ("validation", "β=1"), ("cal+val", "β=1")]:
    for met in ["tvd", "w1", "mode"]:
        t = metrics[(metrics.split == split) & (metrics.metric == met) & (metrics.setting.isin([st, ""]))]
        print(f"\n[{split} · {met} · {st}]\n", t.pivot_table(index="level", columns="model", values="mean", sort=False).reindex(LEVELS).round(3))
print("\n[균등혼합 λ*, validation]\n", metrics[(metrics.split == "validation") & (metrics.setting == "균등혼합 λ*")]
      .pivot_table(index="level", columns="model", values="mean", sort=False).reindex(LEVELS).round(3))
print("\n", paired.round(3).to_string(index=False))
print("\n", line_effect.round(3).to_string(index=False))
print("\n", ladder.round(3).to_string(index=False))
print("\n", mode_radius.round(3).to_string(index=False))
print("\n", pairs_radius.round(3).to_string(index=False))
print("\n", bpr.round(3).to_string(index=False))
print("\n", spread.round(3).to_string())
print("\n", sens_tbl.round(3).to_string(index=False))
print("\n", kcheck.round(3).to_string(index=False))
