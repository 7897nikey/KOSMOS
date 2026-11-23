"""추론 high용 60문항 층화 표본 — 실행 (KOSMOS 루트):
  uv run -q --no-project --python 3.12 --with pandas python research/design_v2_20261112/subsets/build_high60.py

- test 49문항은 넣지 않는다(분석에 못 쓴다). calibration·validation 197문항에서 뽑는다.
- 층: 분할(calibration/validation) × 문항 유형(scale_type). 층 크기에 비례해 배분(최대 나머지 방식).
- 층 안에서는 시드 고정 무작위.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N, SEED = 60, 20261123
it = pd.read_csv(HERE.parent / "items_246.csv", encoding="utf-8-sig")
it = it[it["split_60_20_20"] != "test"].copy()
g = it.groupby(["split_60_20_20", "scale_type"]).size()
q = g / g.sum() * N
alloc = np.floor(q).astype(int)
alloc[(q - alloc).sort_values(ascending=False).index[: N - alloc.sum()]] += 1
rng = np.random.default_rng(SEED)
pick = []
for key, n in alloc.items():
    pool = it[(it["split_60_20_20"] == key[0]) & (it["scale_type"] == key[1])]["var"].sort_values().to_numpy()
    pick += list(rng.choice(pool, n, replace=False))
out = it[it["var"].isin(pick)][["var", "split_60_20_20", "scale_type", "content_domain"]].sort_values("var")
out.to_csv(HERE / "high60.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame({"전체": g, "배분": alloc}))
print(len(out), "문항 →", HERE / "high60.csv")
