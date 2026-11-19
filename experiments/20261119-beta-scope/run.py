"""20261119-beta-scope — 보정 β를 고정하는 단위 비교 (k=50).
실행 (KOSMOS 루트): python experiments/20261119-beta-scope/run.py

p ∝ (개수+1)^β. β는 그리드(0.05~1.5, 0.005)에서 그룹 안 조건들의 calibration 148문항 평균 TVD가 최소인 값.
평가는 validation 49문항. 고정 단위: 조건별 / 문체끼리 공유 / 모델 설정별 / 페르소나 단계별 / 전체 하나.
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


REG = ROOT / "experiments/20261119-register-effect/run.py"
src = REG.read_text(encoding="utf-8").split("rows, info = [], {}")[0]
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "logf = io.StringIO()")
g = {"__file__": str(REG), "io": io}
exec(src, g)
H, RAW, CAL, VAL, EV, BETAS = g["H"], g["RAW"], g["CAL"], g["VAL"], g["EV"], g["BETAS"]
prob, tvd = g["prob"], g["tvd"]
B = 2000
rng = np.random.default_rng(20261119)
LV = {"c0": 0, "L0": 1}
STYLE = {"A": "당신·하십시오체", "B": "당신·해라체", "C": "이 사람·하십시오체", "D": "이 사람·해라체"}
LEVEL = {"c0": "페르소나 없음", "L0": "한국 성인 페르소나"}
SET = {"끔": "DeepSeek 추론 끔", "low": "DeepSeek 추론 low", "haiku": "Haiku 추론 끔"}
NI = len(g["VARS"])

# ── 1. 조건마다 β 격자의 문항별 사람 TVD ──────────────────
G = {}                                                  # key=(설정, 문체, 단계) → [β, 문항]
for s, paths in RAW.items():
    C, n, _ = g["load"]([paths["A"], paths["BCD"]])
    for c in "ABCD":
        for lv, j in LV.items():
            if lv == "c0" and c in "CD":
                continue
            G[(s, c, lv)] = tvd(prob(C[c][j][None], BETAS[:, None, None]), H[j][None])
CONDS = list(G)
assert len(CONDS) == 18


def cname(k):
    return f"{SET[k[0]]} / {LEVEL[k[2]]} / {STYLE[k[1]]}"


def choose(group):
    obj = np.mean([G[k][:, CAL].mean(1) for k in group], axis=0)
    return int(obj.argmin())


SCHEMES = {
    "조건별(지금)": lambda k: [k],
    "문체끼리 공유": lambda k: [q for q in CONDS if q[0] == k[0] and q[2] == k[2]],
    "모델 설정별": lambda k: [q for q in CONDS if q[0] == k[0]],
    "페르소나 단계별": lambda k: [q for q in CONDS if q[2] == k[2]],
    "전체 하나": lambda k: CONDS,
}
BI = {sc: {k: choose(f(k)) for k in CONDS} for sc, f in SCHEMES.items()}
T = {sc: {k: G[k][BI[sc][k]] for k in CONDS} for sc in SCHEMES}          # 문항별 TVD [문항]
BASE = "조건별(지금)"

brow = [dict(조건=cname(k), 설정=SET[k[0]], 페르소나=LEVEL[k[2]], 문체=STYLE[k[1]], **{sc: float(BETAS[BI[sc][k]]) for sc in SCHEMES}) for k in CONDS]
pd.DataFrame(brow).to_csv(out / "betas.csv", index=False, encoding="utf-8-sig")
log("[단위별 선택된 β]")
log(pd.DataFrame(brow).round(3).to_string(index=False))
edge = {sc: sum(BI[sc][k] in (0, len(BETAS) - 1) for k in CONDS) for sc in SCHEMES}
log("β가 격자 끝에 걸린 조건 수:", edge)

# ── 2. 손실 ─────────────────────────────────────────────
vix = np.where(VAL)[0]
bix = rng.integers(0, len(vix), (B, len(vix)))


def boot_mean(v):                                        # v: validation 문항별 값
    m = v.mean()
    bs = v[bix].mean(1)
    return float(m), *np.percentile(bs, [2.5, 97.5])


lrow, lsum = [], []
for sc in SCHEMES:
    for k in CONDS:
        lrow.append(dict(단위=sc, 조건=cname(k), β=float(BETAS[BI[sc][k]]), validation_TVD=float(T[sc][k][VAL].mean()),
                         calibration_TVD=float(T[sc][k][CAL].mean()),
                         validation_손실=float(T[sc][k][VAL].mean() - T[BASE][k][VAL].mean())))
    for scope, ks in [("전체 18개 조건", CONDS)] + [(SET[s], [k for k in CONDS if k[0] == s]) for s in RAW]:
        d = np.mean([T[sc][k][VAL] - T[BASE][k][VAL] for k in ks], axis=0)
        m, lo, hi = boot_mean(d)
        lsum.append(dict(단위=sc, 범위=scope, 조건수=len(ks), β개수=len({(tuple(sorted(SCHEMES[sc](k))), BI[sc][k]) for k in ks}),
                         validation_TVD=float(np.mean([T[sc][k][VAL].mean() for k in ks])),
                         calibration_손실=float(np.mean([T[sc][k][CAL].mean() - T[BASE][k][CAL].mean() for k in ks])),
                         validation_손실=m, 하한=lo, 상한=hi, 허용선_0_005_이하=bool(m <= 0.005)))
pd.DataFrame(lrow).to_csv(out / "loss.csv", index=False, encoding="utf-8-sig")
ls = pd.DataFrame(lsum)
ls.to_csv(out / "loss_summary.csv", index=False, encoding="utf-8-sig")
log("\n[고정 단위별 validation TVD와 조건별 대비 손실 (양수=나빠짐)]")
log(ls.round(4).to_string(index=False))

# ── 3. 효과 30개의 결론이 바뀌는가 ───────────────────────
eff = []
for s in RAW:
    for lv, a, b in (("c0", "A", "B"), ("L0", "A", "B"), ("L0", "C", "D")):
        eff.append(("화계(하십시오체→해라체)", SET[s], f"{LEVEL[lv]} {STYLE[a]}→{STYLE[b]}", (s, a, lv), (s, b, lv)))
    for a, b in (("A", "C"), ("B", "D")):
        eff.append(("인칭(당신→이 사람)", SET[s], f"한국 성인 페르소나 {STYLE[a]}→{STYLE[b]}", (s, a, "L0"), (s, b, "L0")))
    eff.append(("문체 결합", SET[s], f"한국 성인 페르소나 {STYLE['A']}→{STYLE['D']}", (s, "A", "L0"), (s, "D", "L0")))
    for c in "AB":
        eff.append(("페르소나(없음→한국 성인)", SET[s], STYLE[c], (s, c, "c0"), (s, c, "L0")))
for lv, cs in (("c0", "AB"), ("L0", "ABCD")):
    for c in cs:
        eff.append(("추론(끔→low)", "DeepSeek", f"{LEVEL[lv]} {STYLE[c]}", ("끔", c, lv), ("low", c, lv)))
assert len(eff) == 30
erow = []
for cat, setting, lab, ka, kb in eff:
    ix = rng.integers(0, len(vix), (B, len(vix)))
    r = dict(분류=cat, 설정=setting, 비교=lab)
    for sc in SCHEMES:
        d = (T[sc][kb] - T[sc][ka])[VAL]
        lo, hi = np.percentile(d[ix].mean(1), [2.5, 97.5])
        r[f"{sc} 차이"], r[f"{sc} 하한"], r[f"{sc} 상한"] = float(d.mean()), float(lo), float(hi)
        r[f"{sc} 유의"] = bool(lo > 0 or hi < 0)
    erow.append(r)
ed = pd.DataFrame(erow)
ed.to_csv(out / "effects.csv", index=False, encoding="utf-8-sig")
chg = []
for _, r in ed.iterrows():
    for sc in list(SCHEMES)[1:]:
        sign = np.sign(r[f"{sc} 차이"]) != np.sign(r[f"{BASE} 차이"])
        sig = r[f"{sc} 유의"] != r[f"{BASE} 유의"]
        chg.append(dict(분류=r["분류"], 설정=r["설정"], 비교=r["비교"], 단위=sc, 조건별_차이=r[f"{BASE} 차이"], 단위_차이=r[f"{sc} 차이"],
                        조건별_유의=r[f"{BASE} 유의"], 단위_유의=r[f"{sc} 유의"], 부호_바뀜=bool(sign), 유의성_바뀜=bool(sig),
                        바뀜=bool(sign or sig)))
cd = pd.DataFrame(chg)
cd.to_csv(out / "effects_changed.csv", index=False, encoding="utf-8-sig")
log("\n[효과 30개 중 조건별과 부호 또는 유의성이 달라지는 효과 수]")
log(cd.groupby("단위")[["부호_바뀜", "유의성_바뀜", "바뀜"]].sum().to_string())
log("\n[조건별 기준 유의한 효과 수:", int(ed[f"{BASE} 유의"].sum()), "/ 30]")
log(cd[cd["바뀜"]][["분류", "설정", "비교", "단위", "조건별_차이", "단위_차이", "조건별_유의", "단위_유의"]].round(4).to_string(index=False))
log("\n[분류별 효과 평균 차이 (validation, 음수=사람에 가까워짐)]")
cols = [f"{sc} 차이" for sc in SCHEMES]
log(ed.groupby("분류")[cols].mean().round(4).to_string())

# ── 4. 옮겨 쓰기: 한 조건을 빼고 고른 β ──────────────────
METHODS = {
    "같은 설정·단계의 다른 문체에서": lambda k: [q for q in CONDS if q[0] == k[0] and q[2] == k[2] and q[1] != k[1]],
    "같은 문체·단계의 다른 모델 설정에서": lambda k: [q for q in CONDS if q[1] == k[1] and q[2] == k[2] and q[0] != k[0]],
    "같은 설정·문체의 다른 단계에서": lambda k: [q for q in CONDS if q[0] == k[0] and q[1] == k[1] and q[2] != k[2]],
}
trow, tsum = [], []
for mn, f in METHODS.items():
    losses = {}
    for k in CONDS:
        grp = f(k)
        if not grp:
            continue
        bi = choose(grp)
        t = G[k][bi]
        losses[k] = t - T[BASE][k]
        trow.append(dict(방식=mn, 조건=cname(k), 원천조건수=len(grp), β=float(BETAS[bi]), 조건별β=float(BETAS[BI[BASE][k]]),
                         validation_TVD=float(t[VAL].mean()), 조건별_validation_TVD=float(T[BASE][k][VAL].mean()),
                         validation_손실=float(losses[k][VAL].mean())))
    for scope, ks in [("전체", list(losses))] + [(SET[s], [k for k in losses if k[0] == s]) for s in RAW]:
        d = np.mean([losses[k][VAL] for k in ks], axis=0)
        m, lo, hi = boot_mean(d)
        tsum.append(dict(방식=mn, 범위=scope, 조건수=len(ks), validation_손실=m, 하한=lo, 상한=hi, 허용선_0_005_이하=bool(m <= 0.005)))
pd.DataFrame(trow).to_csv(out / "transfer.csv", index=False, encoding="utf-8-sig")
ts = pd.DataFrame(tsum)
ts.to_csv(out / "transfer_summary.csv", index=False, encoding="utf-8-sig")
log("\n[옮겨 쓰기: 한 조건을 빼고 고른 β로 잰 validation 손실 (조건별 β 대비)]")
log(ts.round(4).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {f"{s} {k}": {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                      for s, ps in RAW.items() for k, p in ps.items()},
           "params": {"k": 50, "bootstrap": B, "seed": 20261119, "beta_grid": [0.05, 1.5, 0.005], "schemes": list(SCHEMES)},
           "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
