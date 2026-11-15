"""CPU 분석 환경 점검: 데이터·보정 라이브러리가 기존 고정값을 재현하는지 본다. 실행: python -m pytest -q tests/test_silicon.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "research/design_v2_20261112/dataset"
sys.path.insert(0, str(DS))
import silicon as sl  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return sl.load()


def test_test_split_hidden_by_default(df):
    assert "test" not in set(df["split"])
    assert "test" in set(sl.load(include_test=True)["split"])


def test_table_shape(df):
    one = df.drop_duplicates(sl.KEY)
    assert set(df["model"]) == {"exaone", "deepseek", "haiku"}
    assert one.groupby("model").size().eq(12 * 197).all()       # 페르소나 12 × 문항 197
    h = df.groupby(sl.KEY)["human_p"].sum()
    assert np.allclose(h, 1, atol=1e-4)
    assert (df.groupby(sl.KEY)["count"].sum() == one.set_index(sl.KEY)["n_valid"]).all()


def test_calibrated_probs_sum_to_one(df):
    p = sl.calibrate(df, 0.4).groupby(sl.KEY)["p"].sum()
    assert np.allclose(p, 1)


def test_beta_star_matches_frozen(df):
    frozen = json.loads((ROOT / "research/design_v2_20261112/analysis/out_calib/frozen_beta.json").read_text())["beta"]
    d = df[df["model"].isin(["deepseek", "haiku"])]
    got = sl.fit_beta(d)
    for m in ["deepseek", "haiku"]:
        for lv, key in [("c0", "c0"), ("L0", "L0"), ("L1", "L1cell")]:
            assert got[(m, lv)] == pytest.approx(frozen[f"laplace|{m}|{key}"])
