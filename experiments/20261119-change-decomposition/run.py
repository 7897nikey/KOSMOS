"""20261119-change-decomposition — 추론·페르소나·인칭·화계 변화의 종류 분해 (k=50).
실행 (KOSMOS 루트): python experiments/20261119-change-decomposition/run.py

문항마다 조건 X → 조건 Y에 대해  p_Y ≈ 정규화( p_X^τ · exp(δ·점수) )  를 최대우도로 맞춘다.
  τ: 분포 전체의 날카로움(>1 몰림, <1 퍼짐), δ: 순서형 문항에서 높은 코드 쪽으로의 쏠림.
맞춘 뒤 남는 차이를 보기 역할(X의 최빈 보기·양 끝·중간)별로 센다.
표집 잡음 기준선: 두 조건의 개수를 합친 분포에서 각자 표본 수만큼 다시 뽑아 같은 식으로 맞춘 값(NULL_REPS회).
출력: out/comparisons.csv, out/vs_human.csv, out/items.npz, out/run.log, out/run.json
"""
import csv
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


# register-effect/run.py의 데이터 준비·load·prob·tvd를 그대로 쓴다(분석 본문은 실행하지 않음).
REG = ROOT / "experiments/20261119-register-effect/run.py"
src = REG.read_text(encoding="utf-8").split("rows, info = [], {}")[0]
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "logf = io.StringIO()")
g = {"__file__": str(REG), "io": io}
exec(src, g)
H, MASK, CODES, EV, RAW = g["H"], g["MASK"], g["CODES"], g["EV"], g["RAW"]
VARS, share, tvd = g["VARS"], g["share"], g["tvd"]
NEFF = g["d"]["NEFF"][:2]
LV = {"c0": 0, "L0": 1}
items = list(csv.DictReader((ROOT / "research/design_v2_20261112/item_manifest_v2.csv").open(encoding="utf-8-sig")))
assert [r["var"] for r in items] == list(VARS)
ORDINAL = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in items])
KOPT = MASK.sum(1)
FREE_DELTA = ORDINAL & (KOPT >= 3)                      # δ를 맞추는 문항 (이진·명목형은 τ만)

K, B, NULL_REPS, MIN_N = 50, 2000, 50, 20
rng = np.random.default_rng(20261119)
TAU = np.round(np.exp(np.arange(np.log(0.05), np.log(3.0) + 1e-9, 0.05)), 4)   # 로그 간격 0.05, 0.05~3.0
DEL = np.round(np.arange(-2.0, 2.0001, 0.1), 2)
ID0 = int(np.argmin(np.abs(DEL)))
GRID_EDGE_TAU = (TAU[0], TAU[-1])

# 점수: 문항 안에서 표준화한 보기 코드 (보기 순서 자리는 코드 순)
CODEZ = np.zeros(CODES.shape)
for i in range(len(VARS)):
    c = CODES[i, MASK[i]]
    CODEZ[i, MASK[i]] = (c - c.mean()) / (c.std() or 1.0)

STYLE = {"A": "당신·하십시오체", "B": "당신·해라체", "C": "이 사람·하십시오체", "D": "이 사람·해라체"}
LEVEL = {"c0": "페르소나 없음", "L0": "한국 성인 페르소나"}
SET = {"끔": "DeepSeek 추론 끔", "low": "DeepSeek 추론 low", "haiku": "Haiku 추론 끔"}

# ── 1. 데이터 ───────────────────────────────────────────
DATA = {}
for s, paths in RAW.items():
    C, n, _ = g["load"]([paths["A"], paths["BCD"]])
    for c in "ABCD":
        for lv, j in LV.items():
            if lv == "c0" and c in "CD":
                continue
            DATA[(s, c, lv)] = C[c][j]


# ── 2. 맞추기 ───────────────────────────────────────────
def fit(CA, CB, sel):
    """CA·CB [문항, 보기] (sel 문항만). 반환: 평활 p_X, τ만 맞춘 분포, τ·δ 맞춘 분포, τ, δ."""
    M, S, dmask = MASK[sel], CODEZ[sel], FREE_DELTA[sel]
    pA = (CA + M) / (CA + M).sum(-1, keepdims=True)
    lp = np.where(M, np.log(np.where(M, pA, 1.0)), 0.0)
    lg = TAU[None, :, None, None] * lp[:, None, None, :] + DEL[None, None, :, None] * S[:, None, None, :]
    lg = np.where(M[:, None, None, :], lg, -np.inf)
    lg = lg - lg.max(-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    logp = np.where(M[:, None, None, :], logp, 0.0)
    ll = (CB[:, None, None, :] * logp).sum(-1)                              # [문항, τ, δ]
    m = np.arange(len(CA))
    bt = ll[:, :, ID0].argmax(1)
    ok = dmask[:, None, None] | (np.arange(len(DEL)) == ID0)[None, None, :]
    flat = np.where(ok, ll, -np.inf).reshape(len(m), -1).argmax(1)
    it, idl = divmod(flat, len(DEL))
    p_tau = np.where(M, np.exp(logp[m, bt, ID0]), 0.0)
    p_full = np.where(M, np.exp(logp[m, it, idl]), 0.0)
    return pA, p_tau, p_full, TAU[it], DEL[idl]


def roles(pA, M, sel):
    """보기 역할 마스크 [문항, 보기]: X에서 최빈 보기 / 양 끝(최빈 제외) / 중간."""
    mode = np.zeros_like(M, bool)
    mode[np.arange(len(pA)), pA.argmax(-1)] = True
    k = M.sum(1)
    ext = np.zeros_like(M, bool)
    ext[:, 0] = True
    ext[np.arange(len(k)), k - 1] = True
    ext &= M & ~mode
    return mode, ext, M & ~mode & ~ext


def metrics(CA, CB, sel):
    M = MASK[sel]
    nB = CB.sum(-1)
    shB = CB / nB[:, None]
    pA, pt, pf, tau, dl = fit(CA, CB, sel)
    dm = FREE_DELTA[sel]
    r = {"TVD": tvd(share(CA), shB), "맞추기 전 TVD": tvd(pA, shB), "τ만 맞춘 뒤 TVD": tvd(pt, shB), "τ·δ 맞춘 뒤 TVD": tvd(pf, shB),
         "ln τ": np.log(tau), "δ": np.where(dm, dl, np.nan), "τ 격자 끝": np.isin(tau, GRID_EDGE_TAU).astype(float)}
    same = (share(CA).argmax(-1) == shB.argmax(-1)).astype(float)               # 최빈 보기가 같은 문항 (결과를 본 뒤 추가)
    r.update({"최빈 보기 바뀜": 1 - same, "같음": same, "ln τ ×같음": r["ln τ"] * same, "맞추기 전 TVD ×같음": r["맞추기 전 TVD"] * same,
              "τ만 맞춘 뒤 TVD ×같음": r["τ만 맞춘 뒤 TVD"] * same, "τ·δ 맞춘 뒤 TVD ×같음": r["τ·δ 맞춘 뒤 TVD"] * same})
    rm = roles(pA, M, sel)
    for nm, mk in zip(("최빈 보기", "양 끝 보기", "중간 보기"), rm):
        r[f"맞추기 전 차이 {nm}"] = np.where(dm, ((shB - pA) * mk).sum(-1), np.nan)
        r[f"맞춘 뒤 잔차 {nm}"] = np.where(dm, ((shB - pf) * mk).sum(-1), np.nan)
    return r


def null_metrics(CA, CB, sel):
    """같은 분포(두 조건을 합친 것)에서 다시 뽑은 표본으로 같은 분석을 NULL_REPS번 한 문항별 평균."""
    pooled = share(CA + CB)
    na, nb = CA.sum(-1).astype(int), CB.sum(-1).astype(int)
    acc = None
    for _ in range(NULL_REPS):
        a = rng.multinomial(na, pooled).astype(float)
        b = rng.multinomial(nb, pooled).astype(float)
        m = metrics(a, b, sel)
        acc = {k: v / NULL_REPS for k, v in m.items()} if acc is None else {k: acc[k] + v / NULL_REPS for k, v in m.items()}
    return acc


# ── 3. 비교 ─────────────────────────────────────────────
comps = []
for s in RAW:
    for c in "AB":
        comps.append(("페르소나", s, (s, c, "c0"), (s, c, "L0")))
    for a, b in (("A", "C"), ("B", "D")):
        comps.append(("인칭", s, (s, a, "L0"), (s, b, "L0")))
    for lv, a, b in (("c0", "A", "B"), ("L0", "A", "B"), ("L0", "C", "D")):
        comps.append(("화계", s, (s, a, lv), (s, b, lv)))
    comps.append(("문체 결합", s, (s, "A", "L0"), (s, "D", "L0")))
for lv, cs in (("c0", "AB"), ("L0", "ABCD")):
    for c in cs:
        comps.append(("추론", "끔→low", ("끔", c, lv), ("low", c, lv)))
for lv in LV:
    comps.append(("모델 바꾸기", "DeepSeek 끔→Haiku", ("끔", "A", lv), ("haiku", "A", lv)))


def name(k):
    s, c, lv = k
    return f"{SET[s]} / {LEVEL[lv]} / {STYLE[c]}"


def boot(obs, nul, f):
    """f(사전, 부트스트랩 문항 인덱스 [B, n]) → [B]. 관측 − 잡음 초과분의 평균·구간."""
    n = len(next(iter(obs.values())))
    ix = rng.integers(0, n, (B, n))
    allx = np.arange(n)[None]
    bo, bn = f(obs, ix), f(nul, ix)
    o, nl = f(obs, allx)[0], f(nul, allx)[0]
    lo, hi = np.percentile(bo - bn, [2.5, 97.5])
    return float(o), float(nl), float(o - nl), float(lo), float(hi)


def mean_of(key):
    def f(d, ix):
        v = d[key][ix]
        return np.nanmean(v, axis=1)
    return f


def explained(key):
    def f(d, ix):
        return 1 - d[key][ix].sum(1) / d["맞추기 전 TVD"][ix].sum(1)
    return f


def ratio(num, den):
    return lambda d, ix: d[num][ix].sum(1) / d[den][ix].sum(1)


def explained_same(key):
    return lambda d, ix: 1 - d[key][ix].sum(1) / d["맞추기 전 TVD ×같음"][ix].sum(1)


STATS = [("변화 크기 TVD", mean_of("TVD")), ("ln τ (날카로움)", mean_of("ln τ")), ("δ (순서 기울기)", mean_of("δ")),
         ("설명 비율 τ만", explained("τ만 맞춘 뒤 TVD")), ("설명 비율 τ·δ", explained("τ·δ 맞춘 뒤 TVD"))]
STATS += [("최빈 보기 바뀐 문항 비율", mean_of("최빈 보기 바뀜")), ("ln τ (최빈 보기 같은 문항)", ratio("ln τ ×같음", "같음")),
          ("설명 비율 τ만 (최빈 보기 같은 문항)", explained_same("τ만 맞춘 뒤 TVD ×같음")),
          ("설명 비율 τ·δ (최빈 보기 같은 문항)", explained_same("τ·δ 맞춘 뒤 TVD ×같음"))]
for nm in ("최빈 보기", "양 끝 보기", "중간 보기"):
    STATS += [(f"맞추기 전 차이 {nm}", mean_of(f"맞추기 전 차이 {nm}")), (f"맞춘 뒤 잔차 {nm}", mean_of(f"맞춘 뒤 잔차 {nm}"))]

rows, store = [], {}
for ci, (grp, scope, kx, ky) in enumerate(comps):
    CA, CB = DATA[kx], DATA[ky]
    sel = EV & (CA.sum(-1) >= MIN_N) & (CB.sum(-1) >= MIN_N)
    CAs, CBs = CA[sel], CB[sel]
    obs, nul = metrics(CAs, CBs, sel), null_metrics(CAs, CBs, sel)
    row = {"분류": grp, "범위": scope, "X": name(kx), "Y": name(ky), "문항 수": int(sel.sum()),
           "순서형(δ) 문항 수": int(FREE_DELTA[sel].sum()), "τ 격자 끝 비율": float(obs["τ 격자 끝"].mean())}
    for sn, f in STATS:
        o, nl, ex, lo, hi = boot(obs, nul, f)
        row.update({f"{sn} 관측": o, f"{sn} 잡음": nl, f"{sn} 초과": ex, f"{sn} 하한": lo, f"{sn} 상한": hi,
                    f"{sn} 유의": bool(lo > 0 or hi < 0)})
    rows.append(row)
    for k, v in obs.items():
        store[f"{ci}|관측|{k}"] = v
    for k, v in nul.items():
        store[f"{ci}|잡음|{k}"] = v
    store[f"{ci}|문항"] = np.where(sel)[0]
    log(f"[{ci + 1}/{len(comps)}] {grp} {scope}: {row['X']} → {row['Y']}  "
        f"TVD 초과 {row['변화 크기 TVD 초과']:+.3f}, ln τ 초과 {row['ln τ (날카로움) 초과']:+.3f}, δ 초과 {row['δ (순서 기울기) 초과']:+.3f}, "
        f"설명 τ만 {row['설명 비율 τ만 초과']:+.2f}, τ·δ {row['설명 비율 τ·δ 초과']:+.2f}")

res = pd.DataFrame(rows)
res.to_csv(out / "comparisons.csv", index=False, encoding="utf-8-sig")
np.savez_compressed(out / "items.npz", **store)

# ── 4. 사람과의 격차: 각 조건 → 사람 분포 (잡음 기준선 없음) ──
hrows = []
for (s, c, lv), CA in DATA.items():
    j = LV[lv]
    sel = EV & (CA.sum(-1) >= MIN_N)
    CB = H[j][sel] * NEFF[j][sel][:, None]
    m = metrics(CA[sel], CB, sel)
    hrows.append({"설정": SET[s], "페르소나": LEVEL[lv], "문체": STYLE[c], "문항 수": int(sel.sum()),
                  "맞추기 전 TVD": m["맞추기 전 TVD"].mean(), "ln τ": m["ln τ"].mean(), "δ": np.nanmean(m["δ"]),
                  "τ만 맞춘 뒤 TVD": m["τ만 맞춘 뒤 TVD"].mean(), "τ·δ 맞춘 뒤 TVD": m["τ·δ 맞춘 뒤 TVD"].mean(),
                  "τ 격자 끝 비율": m["τ 격자 끝"].mean()})
vh = pd.DataFrame(hrows)
vh.to_csv(out / "vs_human.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
log("\n[사람과의 격차: 조건 → 사람 분포 (ln τ<0: 사람보다 모델이 더 몰려 있어 퍼뜨려야 함)]")
log(vh.round(3).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {f"{s} {k}": {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                      for s, ps in RAW.items() for k, p in ps.items()},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "min_n": MIN_N, "seed": 20261119,
                      "tau_grid": [float(TAU[0]), float(TAU[-1]), "log step 0.05"], "delta_grid": [-2.0, 2.0, 0.1], "n_comparisons": len(comps)},
           "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
