"""20261123-register-final — 인칭·화계 효과를 β 규칙·Holm 보정·거부 통제로 확정 (k=50, 신규 수집 없음).
실행 (KOSMOS 루트): uv run --no-project --python 3.12 --with pandas --with numpy python experiments/20261123-register-final/run.py

β 규칙(20261123-beta-subgroup): 설정마다 한국 성인 문체 4개가 β 하나, 페르소나 없음 문체 2개가 β 하나.
β는 calibration 148문항에서 문체 평균 TVD 최소. 주 지표: validation 49문항 보정 후 사람 TVD.
"""
import datetime
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
out = HERE / "out"
out.mkdir(exist_ok=True)
logf = open(out / "run.log", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    logf.write(s + "\n")


# 출처: experiments/20261119-register-effect/run.py (load, prob, tvd, H, RAW, 분할) — beta-scope와 같은 방식으로 불러 쓴다
REG = ROOT / "experiments/20261119-register-effect/run.py"
src = REG.read_text(encoding="utf-8").split("rows, info = [], {}")[0]
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "logf = io.StringIO()")
g = {"__file__": str(REG), "io": io}
exec(src, g)
H, RAW, CAL, VAL, EV, BETAS, MASK = g["H"], g["RAW"], g["CAL"], g["VAL"], g["EV"], g["BETAS"], g["MASK"]
prob, tvd, load, share = g["prob"], g["tvd"], g["load"], g["share"]
B = 2000
rng = np.random.default_rng(20261123)
NAME = {"A": "원문(당신·하십시오체)", "B": "당신·해라체", "C": "이 사람·하십시오체", "D": "이 사람·해라체"}
SET = {"끔": "DeepSeek Eco 추론 끔", "low": "DeepSeek Eco 추론 low", "haiku": "Haiku 5.5 추론 끔"}
GROUP = {0: "AB", 1: "ABCD"}                    # 단계별로 β를 공유하는 문체
top = lambda x: np.where(MASK, x, -np.inf).argmax(-1)

# ── 1. 개수, β 규칙, 문항별 TVD ─────────────────────────
DATA, T, T1, BETA, INV = {}, {}, {}, {}, {}
for s, paths in RAW.items():
    C, n, _ = load([paths["A"], paths["BCD"]])
    DATA[s] = C
    for j, cs in GROUP.items():
        grid = {c: tvd(prob(C[c][j][None], BETAS[:, None, None]), H[j][None]) for c in cs}
        bi = int(np.mean([grid[c][:, CAL].mean(1) for c in cs], axis=0).argmin())
        BETA[(s, j)] = float(BETAS[bi])
        for c in cs:
            T[(s, j, c)] = grid[c][bi]
            T1[(s, j, c)] = tvd(prob(C[c][j]), H[j])
            INV[(s, j, c)] = 1 - C[c][j].sum(-1) / np.maximum(n[c][j], 1)
bdf = pd.DataFrame([dict(설정=SET[s], 단계=("페르소나 없음", "한국 성인")[j], β=b) for (s, j), b in BETA.items()])
bdf.to_csv(out / "betas.csv", index=False, encoding="utf-8-sig")
log("[β 규칙으로 고른 β]")
log(bdf.to_string(index=False))

CONTRASTS = {
    "인칭(당신→이 사람)": {"C": .5, "D": .5, "A": -.5, "B": -.5},
    "화계(하십시오체→해라체)": {"B": .5, "D": .5, "A": -.5, "C": -.5},
    "상호작용(인칭×화계)": {"D": 1, "B": -1, "C": -1, "A": 1},
}


def contrast(TT, s, w, j=1):
    return sum(v * TT[(s, j, c)] for c, v in w.items())


def boot(x, sel):
    idx = np.where(sel & ~np.isnan(x))[0]
    bs = x[idx[rng.integers(0, len(idx), (B, len(idx)))]].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
    return float(x[idx].mean()), float(lo), float(hi), float(min(p, 1.0)), len(idx)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


# ── 2. 주 가설군 (9개, Holm) ─────────────────────────────
rows = []
for eff, w in CONTRASTS.items():
    for s in RAW:
        m, lo, hi, p, n_ = boot(contrast(T, s, w), VAL)
        rows.append(dict(효과=eff, 설정=SET[s], 차이=m, 하한=lo, 상한=hi, p=p, 문항수=n_))
ed = pd.DataFrame(rows)
ed["p_Holm"] = holm(ed["p"].to_numpy())
ed["유의"] = ed["p_Holm"] < 0.05
kind = {}
for eff, grp in ed.groupby("효과", sort=False):
    sg = np.sign(grp["차이"])
    kind[eff] = "모델 공통" if (sg.nunique() == 1 and grp["유의"].all()) else ("방향만 공통" if sg.nunique() == 1 else "모델 특이")
ed["분류"] = ed["효과"].map(kind)
ed.to_csv(out / "effects.csv", index=False, encoding="utf-8-sig")
log("\n[주 가설군: 한국 성인 2×2, 보정 후 사람 TVD(validation 49), Holm 9개]")
log(ed.round(4).to_string(index=False))

# ── 3. 탐색 ─────────────────────────────────────────────
xrows = []


def xadd(kind_, eff, s, x, sel):
    m, lo, hi, p, n_ = boot(x, sel)
    xrows.append(dict(분석=kind_, 효과=eff, 설정=SET[s], 차이=m, 하한=lo, 상한=hi, 구간_0_제외=bool(lo > 0 or hi < 0), 문항수=n_))


for s in RAW:
    for eff, w in CONTRASTS.items():
        xadd("197문항 β 규칙", eff, s, contrast(T, s, w), EV)
        xadd("197문항 β=1", eff, s, contrast(T1, s, w), EV)
    for c in "BCD":
        xadd("단순 대비 validation β 규칙", f"한국 성인 {NAME[c]} − 원문", s, T[(s, 1, c)] - T[(s, 1, "A")], VAL)
        xadd("단순 대비 197문항 β 규칙", f"한국 성인 {NAME[c]} − 원문", s, T[(s, 1, c)] - T[(s, 1, "A")], EV)
    xadd("단순 대비 validation β 규칙", "페르소나 없음 당신·해라체 − 원문", s, T[(s, 0, "B")] - T[(s, 0, "A")], VAL)
    xadd("단순 대비 197문항 β 규칙", "페르소나 없음 당신·해라체 − 원문", s, T[(s, 0, "B")] - T[(s, 0, "A")], EV)
xd = pd.DataFrame(xrows)
xd.to_csv(out / "explore.csv", index=False, encoding="utf-8-sig")
log("\n[탐색 (보정 없음)]")
log(xd.round(4).to_string(index=False))

# 거부 민감도
rrows = []
for s in RAW:
    inv = np.stack([INV[(s, 1, c)] for c in "ABCD"])
    keep = (inv <= 0.05).all(0)
    rrows.append(dict(설정=SET[s], **{f"무효율 {NAME[c]}": float(INV[(s, 1, c)][EV].mean()) for c in "ABCD"},
                      남은_validation=int((keep & VAL).sum()), 남은_197=int((keep & EV).sum())))
    for eff, w in CONTRASTS.items():
        m, lo, hi, p, n_ = boot(contrast(T, s, w), VAL & keep)
        m2, lo2, hi2, _, _ = boot(contrast(T, s, w), VAL)
        rrows.append(dict(설정=SET[s], 효과=eff, 전체_차이=m2, 전체_하한=lo2, 전체_상한=hi2, 무효5이하_차이=m, 하한=lo, 상한=hi, 문항수=n_))
rd = pd.DataFrame(rrows)
rd.to_csv(out / "refusal.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 거부 민감도 — 한국 성인 문체 4개 모두 무효율 ≤5%인 문항만]")
log(rd.round(4).to_string(index=False))

# 최빈 보기
mrows = []
for s in RAW:
    hm = top(H[1])
    base = top(DATA[s]["A"][1])
    for c in "BCD":
        mc = top(DATA[s][c][1])
        ch = (mc != base) & EV
        mrows.append(dict(설정=SET[s], 문체=NAME[c], 최빈_바뀐_비율=float(ch.sum() / EV.sum()),
                          바뀐_문항에서_새_최빈이_사람과_같음=float((mc[ch] == hm[ch]).mean()) if ch.any() else np.nan,
                          바뀐_문항에서_원문_최빈이_사람과_같음=float((base[ch] == hm[ch]).mean()) if ch.any() else np.nan))
md = pd.DataFrame(mrows)
md.to_csv(out / "mode.csv", index=False, encoding="utf-8-sig")
log("\n[탐색: 원문 대비 최빈 보기가 바뀐 문항 (197문항, 한국 성인)]")
log(md.round(3).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {f"{s} {k}": {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                      for s, ps in RAW.items() for k, p in ps.items()},
           "params": {"k": 50, "bootstrap": B, "seed": 20261123, "beta_rule": "설정 × 페르소나 유무, 문체 공유"},
           "paid_calls": 0, "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
