"""20261119-item-overlap — 인칭·화계·추론이 흔드는 문항의 겹침·성질·방향 (k=50).
실행 (KOSMOS 루트): python experiments/20261119-item-overlap/run.py

1. 비교(조건 X → Y)마다 문항별: TVD, 잡음 95% 기준 '잡음 밖 흔들림', 최빈 보기 뒤바뀜, 사람 최빈 보기 일치, 점수 이동
2. 대표 비교 7개 사이 흔들리는 문항의 겹침 (초기하 검정, 독립 가정 모의, 스피어만)
3. 흔들림과 문항 성질(사람 1등 비율, 보기 수, 척도 종류, 주제, 민감도, 바람직한 쪽)
4. 최빈 보기가 바뀐 문항에서 사람 쪽/반대쪽, 순서형 문항의 이동 방향, 방향의 일관성
"""
import csv
import datetime
import hashlib
import io
import json
import subprocess
from math import comb
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
H, MASK, CODES, EV, RAW = g["H"], g["MASK"], g["CODES"], g["EV"], g["RAW"]
VARS, share, tvd = list(g["VARS"]), g["share"], g["tvd"]
LV = {"c0": 0, "L0": 1}
man = list(csv.DictReader((ROOT / "research/design_v2_20261112/item_manifest_v2.csv").open(encoding="utf-8-sig")))
assert [r["var"] for r in man] == VARS
NI = len(VARS)
ORDINAL = np.array([r["scale_type"] in ("ordinal_monotone", "binary") for r in man])
KOPT = MASK.sum(1)
ORD = ORDINAL & (KOPT >= 3)                              # 점수 이동을 보는 순서형 문항
CODEZ = np.zeros(CODES.shape)
for i in range(NI):
    c = CODES[i, MASK[i]]
    CODEZ[i, MASK[i]] = (c - c.mean()) / (c.std() or 1.0)

K, B, NULL_REPS, MIN_N, SIM = 50, 2000, 200, 20, 1000
rng = np.random.default_rng(20261119)
STYLE = {"A": "당신·하십시오체", "B": "당신·해라체", "C": "이 사람·하십시오체", "D": "이 사람·해라체"}
LEVEL = {"c0": "페르소나 없음", "L0": "한국 성인 페르소나"}
SET = {"끔": "DeepSeek 추론 끔", "low": "DeepSeek 추론 low", "haiku": "Haiku 추론 끔"}

DATA = {}
for s, paths in RAW.items():
    C, n, _ = g["load"]([paths["A"], paths["BCD"]])
    for c in "ABCD":
        for lv, j in LV.items():
            if not (lv == "c0" and c in "CD"):
                DATA[(s, c, lv)] = C[c][j]

# ── 비교 목록 (조작, 설정, X, Y) ────────────────────────
comps = []
for s in RAW:
    comps += [("인칭", s, (s, "A", "L0"), (s, "C", "L0")), ("인칭", s, (s, "B", "L0"), (s, "D", "L0")),
              ("화계", s, (s, "A", "c0"), (s, "B", "c0")), ("화계", s, (s, "A", "L0"), (s, "B", "L0")),
              ("화계", s, (s, "C", "L0"), (s, "D", "L0"))]
for lv, cs in (("c0", "AB"), ("L0", "ABCD")):
    comps += [("추론", "끔→low", ("끔", c, lv), ("low", c, lv)) for c in cs]
REP = [i for i, (m, s, x, y) in enumerate(comps)
       if (m, x[1], y[1], x[2]) in {("인칭", "A", "C", "L0"), ("화계", "A", "B", "L0"), ("추론", "A", "A", "L0")}]
assert len(REP) == 7


def nm(k):
    return f"{SET[k[0]]} / {LEVEL[k[2]]} / {STYLE[k[1]]}"


def label(i):
    m, s, x, y = comps[i]
    return f"{m} {s} {LEVEL[x[2]]} {STYLE[x[1]]}→{STYLE[y[1]]}" if m != "추론" else f"추론 끔→low {LEVEL[x[2]]} {STYLE[x[1]]}"


MAN = {
    "topic": np.array([r["topic"] for r in man]), "scale": np.array([r["scale_type"] for r in man]),
    "desir": np.array([r["desirable_end"] for r in man]), "sens": np.array([float(r["sensitivity_level"]) for r in man]),
}


def mode_sets(sh):
    return sh == sh.max(-1, keepdims=True)


# ── 1. 비교마다 문항별 값 ────────────────────────────────
rec = []                                                   # 문항×비교 행
cmp_arr = {}
for ci, (m, s, kx, ky) in enumerate(comps):
    CA, CB = DATA[kx], DATA[ky]
    j = LV[kx[2]]
    sel = EV & (CA.sum(-1) >= MIN_N) & (CB.sum(-1) >= MIN_N)
    shA, shB = share(CA), share(CB)
    obs = tvd(shA, shB)
    pooled = share(CA + CB)
    na, nb = CA.sum(-1).astype(int), CB.sum(-1).astype(int)
    nulls = np.full((NULL_REPS, NI), np.nan)
    idx = np.where(sel)[0]
    for r_ in range(NULL_REPS):
        a = rng.multinomial(na[idx], pooled[idx]) / na[idx][:, None]
        b = rng.multinomial(nb[idx], pooled[idx]) / nb[idx][:, None]
        nulls[r_, idx] = tvd(a, b)
    frag = obs > np.nanpercentile(nulls, 95, axis=0)
    excess = obs - np.nanmean(nulls, axis=0)
    flip = ~(mode_sets(shA) & mode_sets(shB)).any(-1)
    hm = H[j].argmax(-1)
    ma, mb = shA.argmax(-1), shB.argmax(-1)
    ra, rb = ma == hm, mb == hm
    sA, sB = (shA * CODEZ).sum(-1), (shB * CODEZ).sum(-1)
    shift = np.where(ORD, sB - sA, np.nan)
    toward_desir = np.where(ORD & np.isin(MAN["desir"], ("first", "last")),
                            shift * np.where(MAN["desir"] == "last", 1.0, -1.0), np.nan)
    newhigher = np.where(flip & ORD, (CODES[np.arange(NI), mb] > CODES[np.arange(NI), ma]).astype(float), np.nan)
    th = tvd(shB, H[j]) - tvd(shA, H[j])
    cmp_arr[ci] = dict(sel=sel, frag=frag & sel, excess=excess, flip=flip & sel, ra=ra, rb=rb, shift=shift,
                       toward=toward_desir, newhigher=newhigher, dth=th, obs=obs)
    for i in idx:
        rec.append(dict(비교번호=ci + 1, 비교=label(ci), 조작=m, 문항=VARS[i], 잡음밖흔들림=bool(frag[i]), 최빈뒤바뀜=bool(flip[i]),
                        TVD=obs[i], TVD초과=excess[i], 사람최빈_X일치=bool(ra[i]), 사람최빈_Y일치=bool(rb[i]),
                        점수이동=shift[i], 사람TVD변화=th[i]))
    log(f"[{ci + 1}/{len(comps)}] {label(ci)}: 문항 {int(sel.sum())}, 잡음 밖 흔들림 {frag[sel].mean():.3f}, 최빈 뒤바뀜 {flip[sel].mean():.3f}")
pd.DataFrame(rec).to_csv(out / "items.csv", index=False, encoding="utf-8-sig")


# ── 2. 겹침 ─────────────────────────────────────────────
def hyper_p(n11, n1, n2, N):
    return sum(comb(n1, x) * comb(N - n1, n2 - x) for x in range(n11, min(n1, n2) + 1)) / comb(N, n2)


def boot_rho(a, b):
    n = len(a)
    ra_, rb_ = pd.Series(a).rank().values, pd.Series(b).rank().values
    ix = rng.integers(0, n, (1000, n))
    x, y = ra_[ix], rb_[ix]
    x, y = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    rho = (x * y).sum(1) / np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))
    return float(np.corrcoef(ra_, rb_)[0, 1]), *np.percentile(rho, [2.5, 97.5])


ov = []
for p in range(len(comps)):
    for q in range(p + 1, len(comps)):
        com = cmp_arr[p]["sel"] & cmp_arr[q]["sel"]
        f1, f2 = cmp_arr[p]["frag"][com], cmp_arr[q]["frag"][com]
        N, n1, n2, n11 = int(com.sum()), int(f1.sum()), int(f2.sum()), int((f1 & f2).sum())
        exp_ = n1 * n2 / N
        rho = boot_rho(cmp_arr[p]["excess"][com], cmp_arr[q]["excess"][com])
        ov.append(dict(비교1=label(p), 비교2=label(q), 대표=(p in REP and q in REP), 공통문항=N, 흔들림1=n1, 흔들림2=n2, 겹침=n11,
                       기대겹침=exp_, 비율=n11 / exp_ if exp_ else np.nan, p초기하=hyper_p(n11, n1, n2, N),
                       스피어만=rho[0], 스피어만하한=rho[1], 스피어만상한=rho[2],
                       같은조작=comps[p][0] == comps[q][0], 같은설정=comps[p][1] == comps[q][1]))
ovd = pd.DataFrame(ov)
ovd.to_csv(out / "overlap.csv", index=False, encoding="utf-8-sig")
rp = ovd[ovd["대표"]]
log("\n[대표 7개 사이 흔들리는 문항 겹침]")
log(rp[["비교1", "비교2", "공통문항", "흔들림1", "흔들림2", "겹침", "기대겹침", "비율", "p초기하", "스피어만", "스피어만하한", "스피어만상한"]].round(3).to_string(index=False))
log(f"비율 1.5 이상이고 p<0.05인 쌍: {int(((rp['비율'] >= 1.5) & (rp['p초기하'] < 0.05)).sum())} / {len(rp)}")

# 7개 중 몇 개에서 흔들리나 — 독립 가정 모의와 비교
common = np.all([cmp_arr[i]["sel"] for i in REP], axis=0)
F = np.array([cmp_arr[i]["frag"][common] for i in REP])             # [7, 문항]
cnt = F.sum(0)
sim = np.zeros((SIM, 8))
for r_ in range(SIM):
    c_ = sum(rng.permutation(F[k]).astype(int) for k in range(7))
    sim[r_] = np.bincount(c_, minlength=8)
rows = []
for k in range(8):
    obs_k = int((cnt == k).sum())
    rows.append(dict(흔들린_비교수=k, 관측_문항수=obs_k, 독립가정_기대=sim[:, k].mean(), 모의하한=np.percentile(sim[:, k], 2.5),
                     모의상한=np.percentile(sim[:, k], 97.5)))
sd = pd.DataFrame(rows)
sd.to_csv(out / "overlap_summary.csv", index=False, encoding="utf-8-sig")
tail_obs = int((cnt >= 5).sum())
tail_p = float((sim[:, 5:].sum(1) >= tail_obs).mean())
log(f"\n[대표 7개 모두 잡음 밖 판정을 받은 공통 문항 {int(common.sum())}개 중, 몇 개의 비교에서 흔들렸나]")
log(sd.round(2).to_string(index=False))
log(f"5개 이상에서 흔들린 문항: 관측 {tail_obs}, 독립 가정 평균 {sim[:, 5:].sum(1).mean():.1f}, 모의 p={tail_p:.3f}")
always = [VARS[i] for i in np.where(common)[0][cnt >= 5]]
log("자주 흔들리는 문항:", always)

# ── 3. 문항 성질 ────────────────────────────────────────
fr_item = np.full(NI, np.nan)
cnt_item = np.zeros(NI)
sum_item = np.zeros(NI)
for i in REP:
    s_ = cmp_arr[i]["sel"]
    sum_item[s_] += cmp_arr[i]["frag"][s_]
    cnt_item[s_] += 1
valid = cnt_item > 0
fr_item[valid] = sum_item[valid] / cnt_item[valid]
htop = H[1].max(-1)
terc = lambda v: np.digitize(v, np.quantile(v[valid], [1 / 3, 2 / 3]))
groups = {"사람 1등 비율": (terc(htop), {0: "낮음(하위 1/3)", 1: "중간", 2: "높음(상위 1/3)"}),
          "민감도": (terc(MAN["sens"]), {0: "낮음(하위 1/3)", 1: "중간", 2: "높음(상위 1/3)"}),
          "보기 수": (np.array(KOPT), {k: f"{k}개" for k in sorted(set(KOPT))}),
          "척도 종류": (MAN["scale"], {k: k for k in sorted(set(MAN["scale"]))}),
          "주제": (MAN["topic"], {k: k for k in sorted(set(MAN["topic"]))}),
          "바람직한 쪽": (MAN["desir"], {k: k for k in sorted(set(MAN["desir"]))})}
ix = rng.integers(0, NI, (B, NI))
prop_rows = []
for gname, (lab, names) in groups.items():
    for key, nmk in names.items():
        mk = valid & (lab == key)
        if mk.sum() == 0:
            continue
        w = mk[ix].astype(float)
        bm = (np.nan_to_num(fr_item)[ix] * w).sum(1) / np.maximum(w.sum(1), 1)
        prop_rows.append(dict(성질=gname, 구간=nmk, 문항수=int(mk.sum()), 흔들림비율=float(fr_item[mk].mean()),
                              하한=float(np.percentile(bm, 2.5)), 상한=float(np.percentile(bm, 97.5))))
    if gname in ("사람 1등 비율", "민감도"):                       # 순서 있는 구간: 상위 − 하위
        hi, lo = valid & (lab == 2), valid & (lab == 0)
        wh, wl = hi[ix].astype(float), lo[ix].astype(float)
        d_ = (np.nan_to_num(fr_item)[ix] * wh).sum(1) / wh.sum(1) - (np.nan_to_num(fr_item)[ix] * wl).sum(1) / wl.sum(1)
        prop_rows.append(dict(성질=gname, 구간="상위 1/3 − 하위 1/3", 문항수=int(hi.sum() + lo.sum()),
                              흔들림비율=float(fr_item[hi].mean() - fr_item[lo].mean()),
                              하한=float(np.percentile(d_, 2.5)), 상한=float(np.percentile(d_, 97.5))))
pr = pd.DataFrame(prop_rows)
pr["유의"] = (pr["하한"] > 0) | (pr["상한"] < 0)
pr.loc[~pr["구간"].str.contains("−"), "유의"] = np.nan
pr.to_csv(out / "properties.csv", index=False, encoding="utf-8-sig")
log(f"\n[흔들림 비율과 문항 성질 (대표 7개 비교의 문항별 평균, 전체 {fr_item[valid].mean():.3f})]")
log(pr.round(3).to_string(index=False))

# ── 4. 방향 ─────────────────────────────────────────────
def bci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), *np.percentile(bs, [2.5, 97.5]), len(x)


hrows, srows = [], []
for ci in range(len(comps)):
    a = cmp_arr[ci]
    sel, fl, ra_, rb_ = a["sel"], a["flip"], a["ra"], a["rb"]
    net = np.where(sel, rb_.astype(float) - ra_.astype(float), np.nan)        # +1: Y가 사람 쪽, −1: X가 사람 쪽
    m_, lo_, hi_, n_ = bci(net[sel])
    f = fl & sel
    hrows.append(dict(비교번호=ci + 1, 비교=label(ci), 문항수=n_, 최빈뒤바뀜=int(f.sum()),
                      뒤바뀜_X맞음Y틀림=int((ra_ & ~rb_ & f).sum()), 뒤바뀜_X틀림Y맞음=int((~ra_ & rb_ & f).sum()),
                      뒤바뀜_둘다틀림=int((~ra_ & ~rb_ & f).sum()),
                      최빈일치_X=float(ra_[sel].mean()), 최빈일치_Y=float(rb_[sel].mean()),
                      순변화=m_, 순변화하한=lo_, 순변화상한=hi_, 순변화유의=bool(lo_ > 0 or hi_ < 0),
                      사람TVD변화_흔들린문항=float(np.nanmean(a["dth"][a["frag"]])) if a["frag"].any() else np.nan,
                      사람TVD변화_안흔들린문항=float(np.nanmean(a["dth"][sel & ~a["frag"]]))))
    sh, tw, nh = a["shift"][sel], a["toward"][sel], a["newhigher"][sel]
    ms, ls, hs, ns = bci(sh)
    mt, lt, ht, nt_ = bci(np.where(np.isnan(tw), np.nan, (tw > 0).astype(float) + 0.5 * (tw == 0)))
    mn, ln_, hn, nn_ = bci(nh)
    srows.append(dict(비교번호=ci + 1, 비교=label(ci), 순서형문항=ns, 점수이동평균=ms, 점수이동하한=ls, 점수이동상한=hs,
                      점수이동유의=bool(ls > 0 or hs < 0), 바람직한쪽문항=nt_, 바람직한쪽이동비율=mt, 바람직한쪽하한=lt, 바람직한쪽상한=ht,
                      바람직한쪽유의=bool(lt > 0.5 or ht < 0.5), 뒤바뀐순서형문항=nn_, 새최빈이_더높은코드_비율=mn,
                      새최빈하한=ln_, 새최빈상한=hn, 새최빈유의=bool(ln_ > 0.5 or hn < 0.5)))
hd, sdf = pd.DataFrame(hrows), pd.DataFrame(srows)
hd.to_csv(out / "human_direction.csv", index=False, encoding="utf-8-sig")
sdf.to_csv(out / "shift_direction.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 280)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
log("\n[최빈 보기 뒤바뀜: 사람 쪽인가 (순변화>0이면 Y가 사람 최빈 보기에 더 많이 맞음)]")
log(hd[["비교", "문항수", "최빈뒤바뀜", "뒤바뀜_X맞음Y틀림", "뒤바뀜_X틀림Y맞음", "뒤바뀜_둘다틀림", "최빈일치_X", "최빈일치_Y", "순변화", "순변화하한", "순변화상한",
        "사람TVD변화_흔들린문항", "사람TVD변화_안흔들린문항"]].round(3).to_string(index=False))
log("\n[순서형 문항의 이동 방향]")
log(sdf[["비교", "순서형문항", "점수이동평균", "점수이동하한", "점수이동상한", "바람직한쪽이동비율", "바람직한쪽하한", "바람직한쪽상한",
         "뒤바뀐순서형문항", "새최빈이_더높은코드_비율", "새최빈하한", "새최빈상한"]].round(3).to_string(index=False))

# 방향 일관성: 대표 7개의 문항별 점수 이동 상관
crow = []
for p in REP:
    for q in REP:
        if q <= p:
            continue
        com = cmp_arr[p]["sel"] & cmp_arr[q]["sel"] & ORD
        r_, lo_, hi_ = boot_rho(cmp_arr[p]["shift"][com], cmp_arr[q]["shift"][com])
        crow.append(dict(비교1=label(p), 비교2=label(q), 문항수=int(com.sum()), 스피어만=r_, 하한=lo_, 상한=hi_,
                         유의=bool(lo_ > 0 or hi_ < 0), 같은조작=comps[p][0] == comps[q][0], 같은설정=comps[p][1] == comps[q][1]))
cr = pd.DataFrame(crow)
cr.to_csv(out / "shift_corr.csv", index=False, encoding="utf-8-sig")
log("\n[대표 7개의 문항별 점수 이동 상관 (같은 문항이 같은 방향으로 움직이나)]")
log(cr.round(3).to_string(index=False))
logf.close()

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")) or bool(git("ls-files", "--others", "--exclude-standard", str(HERE.relative_to(ROOT)))),
           "script_sha256_16": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest()[:16],
           "inputs": {f"{s} {k}": {"path": str(p.relative_to(ROOT)), "sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
                      for s, ps in RAW.items() for k, p in ps.items()},
           "params": {"k": K, "bootstrap": B, "null_reps": NULL_REPS, "sim_reps": SIM, "min_n": MIN_N, "seed": 20261119,
                      "n_comparisons": len(comps), "representative": [label(i) for i in REP]},
           "include_test": False}, open(out / "run.json", "w"), ensure_ascii=False, indent=1)
