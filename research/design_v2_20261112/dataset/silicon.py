"""KOSMOS 분석 최소 라이브러리: pandas + numpy만 쓴다. CPU로 수 초 안에 돈다.

    import silicon as sl
    df = sl.load()                       # long DataFrame (test 49문항은 기본 제외)
    df = sl.calibrate(df, beta=0.35)     # 열 p 추가: 보정 확률
    sl.score(df)                         # 모델×페르소나×문항 TVD
    sl.fit_beta(df)                      # calibration 문항에서 β* 선택
    sl.summary(df, beta)                 # 모델 × 단계(c0/L0/L1) 평균 TVD

보정식: p_k = (count_k + α)^β / Σ_j (count_j + α)^β   (α=1 라플라스). β=1이면 평활만, β<1이면 분포를 편다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 1.0
BETAS = np.round(np.arange(0.05, 1.5001, 0.005), 3)
KEY = ["model", "persona", "var"]
LEVELS = ["c0", "L0", "L1"]


def load(path=None, include_test=False):
    """test 문항은 사전등록상 k=100이 모인 뒤 한 번만 연다. 기본은 제외한다."""
    df = pd.read_csv(path or Path(__file__).with_name("kosmos_long.csv.gz"))
    return df if include_test else df[df["split"] != "test"].reset_index(drop=True)


def calibrate(df, beta=1.0, alpha=ALPHA):
    """열 p(보정 확률)를 붙인다. beta는 숫자 하나 또는 {(model, level): β} 딕셔너리."""
    df = df.copy()
    b = df.apply(lambda r: beta[(r["model"], r["level"])], axis=1) if isinstance(beta, dict) else beta
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


def fit_beta(df, split="calibration", betas=BETAS, by=("model", "level")):
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
    return {k: v[0] for k, v in res.items()}


def summary(df, beta=1.0, split=None):
    d = df if split is None else df[df["split"] == split]
    return level_mean(score(calibrate(d, beta)))


def uniform_baseline(df):
    d = df.assign(p=1 / df["n_options"])
    return level_mean(score(d))
