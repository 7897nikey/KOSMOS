"""20261123-beta-subgroup — 보정 β를 묶는 단위를 성별×연령 10셀까지 넓혀 비교.
실행 (KOSMOS 루트): uv run --no-project --python 3.12 --with pandas --with numpy python experiments/20261123-beta-subgroup/run.py

p ∝ (개수+1)^β. β는 격자(0.05~1.5, 0.005)에서 그룹의 calibration 148문항 목적함수가 최소인 값.
목적함수: 그룹 안 (설정, 단계)를 같은 가중치로 평균, 10셀 안은 pop_share 가중. 평가는 validation 49문항.
단위: 셀마다 / 단계별 / 페르소나 유무 / 설정별 / 전체 하나.
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
DESIGN = ROOT / "research/design_v2_20261112"
NPZ = DESIGN / "analysis/out_final_k100/counts_k100.npz"
PROMPTS = DESIGN / "prompts_v2.csv"
ITEMS = DESIGN / "items_246.csv"
CELLS = DESIGN / "cells.csv"
RAW = {"eco_off": ROOT / "raw/v2/deepseek_eco_p1.jsonl", "eco_low": ROOT / "raw/v2/deepseek_eco_low_p1.jsonl"}
K_ECO, B, NOISE_REPS = 50, 2000, 200
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
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
PERSONAS, VARS, MODELS = list(d["personas"]), list(d["vars"]), list(d["models"])
MASK = ~np.isnan(CODES)
EV, CAL, VAL = SPLIT != "test", SPLIT == "calibration", SPLIT == "validation"
NI, NJ = len(VARS), len(PERSONAS)
items = {r["var"]: r for r in csv.DictReader(ITEMS.open(encoding="utf-8-sig"))}
FOLD = np.array([int(items[v]["group_cv_fold"]) for v in VARS])
cells = {r["cell_id"]: float(r["pop_share"]) for r in csv.DictReader(CELLS.open(encoding="utf-8-sig"))}
L1J = list(range(2, 12))
W = np.zeros(NJ)
W[0] = W[1] = 1.0
W[L1J] = np.array([cells[PERSONAS[j]] for j in L1J])
W[L1J] /= W[L1J].sum()
LEVEL_OF = {0: "c0", 1: "L0", **{j: "L1" for j in L1J}}
LEVEL_NAME = {"c0": "페르소나 없음", "L0": "한국 성인", "L1": "성별×연령 10셀"}
probes = {r["probe_id"]: r for r in csv.DictReader(PROMPTS.open(encoding="utf-8-sig")) if r["phase"] == "1"}


def load_counts(path):
    """출처: experiments/20261118-reasoning-effect/run.py load_counts (표본 0~49, 중복·api_error 제외)."""
    C = np.zeros((NJ, NI, CODES.shape[1]))
    seen = set()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        key = (r["probe_id"], r["sample_idx"])
        if r.get("invalid_type") == "api_error" or r["sample_idx"] >= K_ECO or key in seen or r["probe_id"] not in probes:
            continue
        seen.add(key)
        p = probes[r["probe_id"]]
        if p["cell_id"] not in PERSONAS:
            continue
        j, i = PERSONAS.index(p["cell_id"]), VARS.index(p["var"])
        if r["valid"]:
            C[j, i, list(CODES[i]).index(float(r["parsed"]))] += 1
    return C


def prob(X, beta=1.0):
    """출처: experiments/20261119-register-effect/run.py prob."""
    x = (X + MASK) ** beta * MASK
    return x / x.sum(-1, keepdims=True)


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


SET = {"exaone": "K-EXAONE 2.5", "deepseek": "DeepSeek Ultrafast 추론 끔", "haiku": "Haiku 5.5 추론 끔",
       "eco_off": "DeepSeek Eco 추론 끔", "eco_low": "DeepSeek Eco 추론 low"}
COUNTS = {m: d["C"][MODELS.index(m)] for m in ("exaone", "deepseek", "haiku")}
for s, p in RAW.items():
    COUNTS[s] = load_counts(p)
SETS = list(SET)
for s in SETS:
    n = COUNTS[s][:, EV].sum(-1)
    log(f"{SET[s]}: 프롬프트당 유효 표본 최소 {n.min():.0f}, 중앙 {np.median(n):.0f}, 최대 {n.max():.0f}")

# G[(s, j)] = [β, 문항] 셀 정답 대비 TVD. GX[(s, j)] = 한국 성인 분포를 셀 j 정답과 비교한 TVD (셀 페르소나 효과용)
G = {(s, j): tvd(prob(COUNTS[s][j][None], BETAS[:, None, None]), H[j][None]) for s in SETS for j in range(NJ)}
GX = {(s, j): tvd(prob(COUNTS[s][1][None], BETAS[:, None, None]), H[j][None]) for s in SETS for j in L1J}
UNITS = [(s, lv) for s in SETS for lv in ("c0", "L0", "L1")]
JS = {"c0": [0], "L0": [1], "L1": L1J}
CONDS = [(s, j) for s in SETS for j in range(NJ)]


def objective(group, sel):
    """group: (s, j) 목록. (s, 단계)마다 pop_share 가중 평균 → 단계끼리 같은 가중치 평균. sel: 쓸 문항 마스크."""
    by = {}
    for s, j in group:
        by.setdefault((s, LEVEL_OF[j]), []).append(j)
    parts = []
    for (s, lv), js in by.items():
        w = W[js] / W[js].sum()
        parts.append(sum(w_ * G[(s, j)][:, sel].mean(1) for w_, j in zip(w, js)))
    return np.mean(parts, axis=0)


SCHEMES = {
    "셀마다": lambda k: [k],
    "단계별": lambda k: [q for q in CONDS if q[0] == k[0] and LEVEL_OF[q[1]] == LEVEL_OF[k[1]]],
    "페르소나 유무": lambda k: [q for q in CONDS if q[0] == k[0] and (q[1] == 0) == (k[1] == 0)],
    "설정별": lambda k: [q for q in CONDS if q[0] == k[0]],
    "전체 하나": lambda k: CONDS,
}
BASE = "셀마다"


def choose_all(sel):
    res = {}
    for sc, f in SCHEMES.items():
        cache = {}
        res[sc] = {}
        for k in CONDS:
            grp = tuple(f(k))
            if grp not in cache:
                cache[grp] = int(objective(grp, sel).argmin())
            res[sc][k] = cache[grp]
    return res


BI = choose_all(CAL)


def unit_vec(s, lv, bis, src=G):
    """(설정, 단계)의 문항별 TVD: 셀 pop_share 가중 평균."""
    js = JS[lv]
    w = W[js] / W[js].sum()
    return sum(w_ * src[(s, j)][bis[(s, j)]] for w_, j in zip(w, js))


T = {sc: {u: unit_vec(*u, BI[sc]) for u in UNITS} for sc in SCHEMES}

# ── 2. 선택된 β ─────────────────────────────────────────
brow = [dict(설정=SET[s], 페르소나=PERSONAS[j], 단계=LEVEL_NAME[LEVEL_OF[j]], **{sc: float(BETAS[BI[sc][(s, j)]]) for sc in SCHEMES})
        for s, j in CONDS]
bdf = pd.DataFrame(brow)
bdf.to_csv(out / "betas.csv", index=False, encoding="utf-8-sig")
log("\n[단위별 선택된 β]")
log(bdf.round(3).to_string(index=False))
log("β가 격자 끝에 걸린 조건 수:", {sc: sum(BI[sc][k] in (0, len(BETAS) - 1) for k in CONDS) for sc in SCHEMES})
cellb = bdf[bdf["단계"] == "성별×연령 10셀"].groupby("설정")["셀마다"].agg(["min", "max", "std"])
log("\n[셀마다 β의 셀 간 범위 (10셀)]")
log(cellb.round(3).to_string())

# ── 3. 손실 ─────────────────────────────────────────────
vix = np.where(VAL)[0]
bix = rng.integers(0, len(vix), (B, len(vix)))


def boot(v, ix=bix, idx=vix):
    m = float(v[idx].mean())
    lo, hi = np.percentile(v[idx][ix].mean(1), [2.5, 97.5])
    return m, float(lo), float(hi)


lrow, lsum = [], []
for sc in SCHEMES:
    for u in UNITS:
        lrow.append(dict(단위=sc, 설정=SET[u[0]], 단계=LEVEL_NAME[u[1]], validation_TVD=float(T[sc][u][VAL].mean()),
                         calibration_TVD=float(T[sc][u][CAL].mean()),
                         validation_손실=float(T[sc][u][VAL].mean() - T[BASE][u][VAL].mean())))
    scopes = [("전체 15칸", UNITS)] + [(LEVEL_NAME[lv], [u for u in UNITS if u[1] == lv]) for lv in ("c0", "L0", "L1")] \
        + [(SET[s], [u for u in UNITS if u[0] == s]) for s in SETS]
    for scope, us in scopes:
        dv = np.mean([T[sc][u] - T[BASE][u] for u in us], axis=0)
        m, lo, hi = boot(dv)
        lsum.append(dict(단위=sc, 범위=scope, 칸수=len(us), β개수=len({tuple(SCHEMES[sc](k)) for k in CONDS}),
                         validation_TVD=float(np.mean([T[sc][u][VAL].mean() for u in us])),
                         validation_손실=m, 하한=lo, 상한=hi, 허용선_0_005_이하=bool(m <= 0.005)))
pd.DataFrame(lrow).to_csv(out / "loss.csv", index=False, encoding="utf-8-sig")
ls = pd.DataFrame(lsum)
ls.to_csv(out / "loss_summary.csv", index=False, encoding="utf-8-sig")
log("\n[단위별 validation TVD와 셀마다 β 대비 손실 (양수=나빠짐)]")
log(ls.round(4).to_string(index=False))

# ── 4. 효과 19개 ────────────────────────────────────────
TX = {sc: {s: unit_vec(s, "L1", {(s, j): BI[sc][(s, 1)] for j in L1J}, GX) for s in SETS} for sc in SCHEMES}
eff = []
for s in SETS:
    eff.append(("페르소나(없음→한국 성인)", SET[s], "", lambda sc, s=s: T[sc][(s, "L0")] - T[sc][(s, "c0")]))
for s in SETS:
    eff.append(("셀 페르소나(한국 성인 분포→10셀 분포)", SET[s], "셀 정답 기준", lambda sc, s=s: T[sc][(s, "L1")] - TX[sc][s]))
for lv in ("c0", "L0", "L1"):
    eff.append(("추론(Eco 끔→low)", "DeepSeek Eco", LEVEL_NAME[lv], lambda sc, lv=lv: T[sc][("eco_low", lv)] - T[sc][("eco_off", lv)]))
for a, b_ in (("haiku", "exaone"), ("deepseek", "exaone"), ("haiku", "deepseek")):
    for lv in ("L0", "L1"):
        eff.append(("모델 비교", f"{SET[a]} − {SET[b_]}", LEVEL_NAME[lv], lambda sc, a=a, b_=b_, lv=lv: T[sc][(a, lv)] - T[sc][(b_, lv)]))
assert len(eff) == 19
erow = []
for cat, setting, lab, fn in eff:
    ix = rng.integers(0, len(vix), (B, len(vix)))
    r = dict(분류=cat, 설정=setting, 비교=lab)
    for sc in SCHEMES:
        m, lo, hi = boot(fn(sc), ix)
        r[f"{sc} 차이"], r[f"{sc} 하한"], r[f"{sc} 상한"], r[f"{sc} 유의"] = m, lo, hi, bool(lo > 0 or hi < 0)
    erow.append(r)
ed = pd.DataFrame(erow)
ed.to_csv(out / "effects.csv", index=False, encoding="utf-8-sig")
chg = []
for _, r in ed.iterrows():
    for sc in list(SCHEMES)[1:]:
        sign = np.sign(r[f"{sc} 차이"]) != np.sign(r[f"{BASE} 차이"])
        sig = r[f"{sc} 유의"] != r[f"{BASE} 유의"]
        chg.append(dict(분류=r["분류"], 설정=r["설정"], 비교=r["비교"], 단위=sc, 셀마다_차이=r[f"{BASE} 차이"], 단위_차이=r[f"{sc} 차이"],
                        셀마다_유의=r[f"{BASE} 유의"], 단위_유의=r[f"{sc} 유의"], 부호_바뀜=bool(sign), 유의성_바뀜=bool(sig), 바뀜=bool(sign or sig)))
cd = pd.DataFrame(chg)
cd.to_csv(out / "effects_changed.csv", index=False, encoding="utf-8-sig")
log("\n[효과 19개 (셀마다 β 기준)]")
log(ed[["분류", "설정", "비교", f"{BASE} 차이", f"{BASE} 하한", f"{BASE} 상한", f"{BASE} 유의"]].round(4).to_string(index=False))
log("\n[셀마다 β와 부호 또는 유의성이 달라지는 효과 수]")
log(cd.groupby("단위", sort=False)[["부호_바뀜", "유의성_바뀜", "바뀜"]].sum().to_string())
log(cd[cd["바뀜"]][["분류", "설정", "비교", "단위", "셀마다_차이", "단위_차이", "셀마다_유의", "단위_유의"]].round(4).to_string(index=False))

# ── 5. 옮겨 쓰기 ────────────────────────────────────────
tsum = []
# (1) 한국 성인 β를 10셀에 그대로
dv = np.mean([unit_vec(s, "L1", {(s, j): int(G[(s, 1)][:, CAL].mean(1).argmin()) for j in L1J}) - T[BASE][(s, "L1")] for s in SETS], axis=0)
tsum.append(dict(방식="한국 성인 β를 10셀에 그대로", 칸수=len(SETS), **dict(zip(("validation_손실", "하한", "상한"), boot(dv)))))
# (2) 다른 설정들의 같은 단계에서 고른 β (10셀은 셀 하나 공유)
diffs = []
for s, lv in UNITS:
    grp = [(q, j) for q in SETS if q != s for j in JS[lv]]
    bi = int(objective(grp, CAL).argmin())
    diffs.append(unit_vec(s, lv, {(s, j): bi for j in JS[lv]}) - T[BASE][(s, lv)])
tsum.append(dict(방식="다른 설정들의 같은 단계 β", 칸수=len(UNITS), **dict(zip(("validation_손실", "하한", "상한"), boot(np.mean(diffs, axis=0))))))
ts = pd.DataFrame(tsum)
ts["허용선_0_005_이하"] = ts["validation_손실"] <= 0.005
ts.to_csv(out / "transfer_summary.csv", index=False, encoding="utf-8-sig")
log("\n[옮겨 쓰기 손실 (셀마다 β 대비)]")
log(ts.round(4).to_string(index=False))

# ── 6. 보조: 5겹 교차검증 (test 제외 197문항) ────────────
cvT = {sc: {u: np.full(NI, np.nan) for u in UNITS} for sc in SCHEMES}
for f in range(5):
    tr, te = EV & (FOLD != f), EV & (FOLD == f)
    bi_f = choose_all(tr)
    for sc in SCHEMES:
        for u in UNITS:
            cvT[sc][u][te] = unit_vec(*u, bi_f[sc])[te]
eix = np.where(EV)[0]
cix = rng.integers(0, len(eix), (B, len(eix)))
crow = []
for sc in SCHEMES:
    dv = np.mean([cvT[sc][u] - cvT[BASE][u] for u in UNITS], axis=0)
    m, lo, hi = boot(dv, cix, eix)
    crow.append(dict(단위=sc, 교차검증_TVD=float(np.mean([cvT[sc][u][EV].mean() for u in UNITS])), 손실=m, 하한=lo, 상한=hi))
cv = pd.DataFrame(crow)
cv.to_csv(out / "cv_summary.csv", index=False, encoding="utf-8-sig")
log("\n[보조: 5겹 교차검증 손실 (셀마다 β 대비, 197문항)]")
log(cv.round(4).to_string(index=False))

# ── 7. 보조: 셀 정답의 표집 잡음 ─────────────────────────
nrow = []
for j in range(NJ):
    vals = []
    for i in vix:
        k_ = MASK[i].sum()
        p = H[j, i, :k_] / H[j, i, :k_].sum()
        n = max(int(NEFF[j, i]), 1)
        vals.append(tvd(rng.multinomial(n, p, size=NOISE_REPS) / n, p).mean())
    nrow.append(dict(페르소나=PERSONAS[j], n_eff_중앙=float(np.median(NEFF[j, vix])), 정답_잡음_TVD=float(np.mean(vals))))
nz = pd.DataFrame(nrow)
nz.to_csv(out / "noise.csv", index=False, encoding="utf-8-sig")
log("\n[보조: 셀 정답을 n_eff명으로 다시 뽑았을 때 TVD (validation 평균)]")
log(nz.round(4).to_string(index=False))
gap = np.mean([abs(T["단계별"][(s, "L1")][VAL].mean() - T[BASE][(s, "L1")][VAL].mean()) for s in SETS])
log(f"10셀: 셀마다 β와 단계별 β의 validation TVD 차이 평균 {gap:.4f} vs 셀 정답 잡음 평균 {nz['정답_잡음_TVD'][2:].mean():.4f}")
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in [NPZ, *RAW.values()]},
           "params": {"k_eco": K_ECO, "bootstrap": B, "seed": 20261123, "beta_grid": [0.05, 1.5, 0.005], "schemes": list(SCHEMES)},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
