"""KOSMOS 분석 최소 라이브러리: pandas + numpy만 쓴다. CPU로 수 초 안에 돈다.

    import silicon as sl
    df = sl.load()                       # long DataFrame (test 49문항은 기본 제외)
    df = sl.calibrate(df, beta=0.35)     # 열 p 추가: 보정 확률
    sl.score(df)                         # 모델×페르소나×문항 TVD
    sl.fit_beta(df)                      # calibration 문항에서 β* 선택
    sl.summary(df, beta)                 # 모델 × 단계(c0/L0/L1) 평균 TVD
    sl.save_run(out_dir, params, tables) # 결과 표 + run.json(커밋·데이터 해시·시각) 기록

보정식: p_k = (count_k + α)^β / Σ_j (count_j + α)^β   (α=1 라플라스). β=1이면 평활만, β<1이면 분포를 편다.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 1.0
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
KEY = ["model", "persona", "var"]
DATA = Path(__file__).with_name("kosmos_long.csv.gz")
LEVELS = ["c0", "L0", "L1"]


def load(path=None, include_test=False):
    """test 문항은 사전등록상 k=100이 모인 뒤 한 번만 연다. 기본은 제외한다."""
    df = pd.read_csv(path or DATA)
    return df if include_test else df[df["split"] != "test"].reset_index(drop=True)


def calibrate(df, beta=1.0, alpha=ALPHA):
    """열 p(보정 확률)를 붙인다. beta는 숫자 하나 또는 {(model, level): β} 딕셔너리."""
    df = df.copy()
    if isinstance(beta, dict):
        b = pd.Series([beta[k] for k in zip(df["model"], df["level"])], index=df.index)
    else:
        b = beta
    x = (df["count"] + alpha) ** b
    df["p"] = x / x.groupby([df[k] for k in KEY]).transform("sum")
    return df


def raw_share(df):
    """보정 없는 원비율 열 r (유효 응답 대비)."""
    return df.assign(r=df["count"] / df["n_valid"].clip(lower=1))


def score(df, col="p"):
    """(model, level, persona, var) 단위 TVD·최빈 보기 확률·엔트로피. 사람 분포는 human_p."""
    g = df.assign(d=(df[col] - df["human_p"]).abs()).groupby(KEY + ["level"], sort=False)
    out = g["d"].sum().mul(0.5).rename("tvd").to_frame()
    out["top_share"] = g[col].max()
    out["entropy"] = df.assign(h=-df[col] * np.log(df[col].where(df[col] > 0, 1))).groupby(
        KEY + ["level"], sort=False)["h"].sum() / np.log(g["n_options"].first())
    out["split"] = g["split"].first()
    return out.reset_index()


def level_mean(s, metric="tvd"):
    """페르소나 → 단계 평균. L1은 10셀을 문항별로 먼저 평균한 뒤 문항 평균 (기존 분석과 같음)."""
    per_item = s.groupby(["model", "level", "var"])[metric].mean()
    return per_item.groupby(["model", "level"]).mean().unstack("level")[LEVELS]


def fit_beta(df, split="calibration", betas=BETAS):
    """split 문항의 평균 TVD를 최소로 하는 β. 반환: {(model, level): β*}."""
    sub = df[df["split"] == split]
    res = {}
    for b in betas:
        m = level_mean(score(calibrate(sub, b)))
        for model in m.index:
            for lv in LEVELS:
                v = m.loc[model, lv]
                if (model, lv) not in res or v < res[(model, lv)][1] - 1e-12:
                    res[(model, lv)] = (b, v)
    return {k: float(v[0]) for k, v in res.items()}


def summary(df, beta=1.0, split=None):
    d = df if split is None else df[df["split"] == split]
    return level_mean(score(calibrate(d, beta)))


def uniform_baseline(df):
    d = df.assign(p=1 / df["n_options"])
    return level_mean(score(d))


# ── 기록 ─────────────────────────────────────────────────
def provenance(exclude=None):
    """결과를 다시 만들 수 있게 하는 최소 정보: 커밋, 작업트리 변경 여부, 데이터 해시, 버전.
    exclude: 변경 여부에서 뺄 경로(이번 실행의 out/ 폴더).
    """
    here = Path(__file__).resolve().parent
    git = lambda *a: subprocess.run(["git", *a], cwd=here, capture_output=True, text=True).stdout.strip()
    top = git("rev-parse", "--show-toplevel")
    spec = []
    if top and exclude:
        rel = Path(exclude).resolve().relative_to(Path(top).resolve())
        spec = [":(top)", f":(exclude,top){rel.as_posix()}"]
    build = Path(__file__).with_name("kosmos_build.json")
    return {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(git("status", "--porcelain", "--", *spec)) if top else None,
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16],
        "data_built_at": json.loads(build.read_text()).get("built_at") if build.exists() else None,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
    }


def save_run(out_dir, params=None, tables=None, notes=""):
    """out_dir에 표(csv)와 run.json을 쓴다. tables: {이름: DataFrame}. β 딕셔너리 같은 키도 문자열로 바꿔 저장한다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in (tables or {}).items():
        t.to_csv(out / f"{name}.csv", float_format="%.6g")
    fix = lambda o: {"|".join(map(str, k)) if isinstance(k, tuple) else k: fix(v) for k, v in o.items()} \
        if isinstance(o, dict) else o
    run = {"params": fix(params or {}), "tables": sorted(tables or {}), "notes": notes, **provenance(exclude=out)}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1, default=str))
    return run
