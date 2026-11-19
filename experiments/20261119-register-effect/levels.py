"""20261119-register-effect 보조 — 조건별 주요 지표의 수준값(차이 아님).
실행 (KOSMOS 루트): python experiments/20261119-register-effect/levels.py

모델 설정 × 페르소나(c0·L0) × 문체(A~D)마다
  사람 TVD(β=1, 197문항), 보정 후 사람 TVD(L0만, β는 calibration에서, validation 49문항),
  최빈값 일치율(197문항), 보기 순서쌍 일치율(197문항).
보기 순서쌍 일치율: 문항마다 사람 비율이 다른 보기 쌍에서 모델 비율의 대소가 같으면 1, 모델이 같으면 0.5, 반대면 0.
문항을 가리지 않고 쌍을 모두 모아 평균한다. (개수+1)^β는 순서를 바꾸지 않으므로 보정과 무관하다.
출력: out/levels.csv, out/levels.log, out/levels_run.json
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
buf = io.StringIO()

# run.py의 데이터 준비·load·prob·tvd를 그대로 쓴다(분석 본문은 실행하지 않음).
src = (HERE / "run.py").read_text(encoding="utf-8").split("rows, info = [], {}")[0]
src = src.replace('logf = open(out / "run.log", "w", encoding="utf-8")', "logf = io.StringIO()")
g = {"__file__": str(HERE / "run.py"), "io": io}
exec(src, g)
H, MASK, EV, CAL, VAL, BETAS = g["H"], g["MASK"], g["EV"], g["CAL"], g["VAL"], g["BETAS"]
prob, share, tvd, top = g["prob"], g["share"], g["tvd"], g["top"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    buf.write(s + "\n")


def pair_agree(R, Hj):
    """R, Hj [문항, 보기]: EV 문항의 사람 비율이 다른 보기 쌍에서 모델 대소 일치율."""
    hit = tot = 0.0
    for i in np.where(EV)[0]:
        k = MASK[i].sum()
        r, h = R[i, :k], Hj[i, :k]
        dr, dh = r[:, None] - r[None, :], h[:, None] - h[None, :]
        sel = np.triu(np.ones((k, k), bool), 1) & (dh != 0)
        hit += np.where(np.sign(dr) == np.sign(dh), 1.0, np.where(dr == 0, 0.5, 0.0))[sel].sum()
        tot += sel.sum()
    return hit / tot, int(tot)


NAME = {"끔": "DeepSeek 추론 끔", "low": "DeepSeek 추론 low", "haiku": "Haiku 추론 끔"}
STYLE = {"A": "당신·하십시오체(원문)", "B": "당신·해라체", "C": "이 사람·하십시오체", "D": "이 사람·해라체"}
rows = []
for setting, paths in g["RAW"].items():
    C, n, _ = g["load"]([paths["A"], paths["BCD"]])
    for lv, j in g["LV"].items():
        for c in "ABCD":
            if lv == "c0" and c in "CD":
                continue
            X = C[c][j]
            row = dict(model=NAME[setting], persona="페르소나 없음" if lv == "c0" else "한국 성인", style=STYLE[c],
                       setting=setting, level=lv, cond=c,
                       tvd_beta1=float(tvd(prob(X), H[j])[EV].mean()),
                       mode_agree=float((top(share(X)) == top(H[j]))[EV].mean()))
            row["pair_agree"], row["n_pairs"] = pair_agree(share(X), H[j])
            if lv == "L0":
                gr = tvd(prob(X[None], BETAS[:, None, None]), H[j][None])
                b = gr[:, CAL].mean(1).argmin()
                row["tvd_calibrated_val"], row["beta"] = float(gr[b][VAL].mean()), float(BETAS[b])
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(out / "levels.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
log(df.drop(columns=["setting", "level", "cond"]).round(4).to_string(index=False))
(out / "levels.log").write_text(buf.getvalue(), encoding="utf-8")

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "inputs": {f"{s} {k}": hashlib.sha256(p.read_bytes()).hexdigest()[:16] for s, ps in g["RAW"].items() for k, p in ps.items()},
           "params": {"k": g["K"], "items_main": "calibration+validation 197", "items_calibrated": "validation 49"},
           "include_test": False}, open(out / "levels_run.json", "w"), ensure_ascii=False, indent=1)
