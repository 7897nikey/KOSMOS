"""원응답(jsonl) → 분석용 long DataFrame 한 장으로 만든다. API 호출 없음.

산출 (이 폴더):
  kosmos_long.csv.gz   모델 × 페르소나 × 문항 × 보기 한 줄 (count, n_valid, n_total, human_p, human_neff, split 포함)
  kosmos_build.json    소스 파일, 표본 수, 생성 시각

실행 (KOSMOS 루트, pandas·pyarrow 필요):
  .venv-nb/bin/python research/design_v2_20261112/dataset/build_dataset.py

EXAONE은 팀원 결과가 모이는 대로 다시 실행하면 갱신된다. 프롬프트마다 k(n_total)가 다를 수 있다.
"""
from __future__ import annotations

import collections
import csv
import datetime
import gzip
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "inventory/kgss_clean.parquet").exists())
DESIGN = ROOT / "research/design_v2_20261112"
HERE = Path(__file__).resolve().parent
PKG = DESIGN / "team_packages/KOSMOS_EXAONE_m1_태우"

items = pd.read_csv(DESIGN / "item_manifest_v2.csv", encoding="utf-8-sig")
items["codes"] = items["codes"].apply(json.loads)
cells = pd.read_csv(DESIGN / "cells.csv", encoding="utf-8-sig").query("level == 'L1'")
probes = {r["probe_id"]: r for r in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig"))
          if r["phase"] == "1"}


# ── 사람 정답지: KGSS 2023, FINALWT 가중 ───────────────────────
kgss = pd.read_parquet(ROOT / "inventory/kgss_clean.parquet").query("YEAR == 2023")
kgss = kgss.assign(age_band=pd.cut(kgss["AGE"], [18, 30, 40, 50, 60, 200], right=False,
                                   labels=["18-29", "30-39", "40-49", "50-59", "60+"]).astype(str))
SEX = {"남자": 1, "여자": 2}
resp = {"c0": kgss, "L0": kgss} | {c.cell_id: kgss[(kgss["SEX"] == SEX[c.sex]) & (kgss["age_band"] == c.age_band)]
                                   for c in cells.itertuples()}
LEVEL = {"c0": "c0", "L0": "L0"} | {c: "L1" for c in cells["cell_id"]}


def human(persona, var, codes):
    d = resp[persona][[var, "FINALWT"]].dropna()
    d = d[d[var].isin(codes)]
    w = d.groupby(var)["FINALWT"].sum().reindex(codes).fillna(0)
    neff = round(d["FINALWT"].sum() ** 2 / (d["FINALWT"] ** 2).sum())
    return (w / w.sum()).to_numpy(), neff


# ── 원응답 읽기 ────────────────────────────────────────────
def lines_of(path: Path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".jsonl"):
                    yield from io.TextIOWrapper(z.open(n), encoding="utf-8")
    elif path.suffix == ".gz":
        yield from io.TextIOWrapper(gzip.open(path), encoding="utf-8")
    else:
        yield from open(path, encoding="utf-8")


def read(paths):
    """(probe_id, sample_idx) 중복은 먼저 읽은 것만 쓴다. api_error는 제외."""
    store = {}
    for p in paths:
        for line in lines_of(p):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:      # 쓰는 중인 마지막 줄
                continue
            if r.get("invalid_type") == "api_error" or r["probe_id"] not in probes:
                continue
            store.setdefault((r["probe_id"], r["sample_idx"]),
                             (bool(r["valid"]), r.get("parsed"), r.get("invalid_type") or ""))
    return store


def first_existing(*cands):
    return next(c for c in cands if c.exists())


SOURCES = {
    "exaone": [*sorted(PKG.glob("KOSMOS_EXAONE_결과_m*_*.zip")), *sorted(PKG.glob("exaone_m*_p1.jsonl"))],
    "deepseek": [first_existing(ROOT / "raw/v2/deepseek_ultrafast_p1.jsonl",
                                ROOT / "results/raw_gz/v2/deepseek_ultrafast_p1.jsonl.gz")],
    "haiku": [first_existing(ROOT / "raw/v2/haiku_p1.jsonl", ROOT / "results/raw_gz/v2/haiku_p1.jsonl.gz")],
}

# ── 집계 ──────────────────────────────────────────────────
item_codes = dict(zip(items["var"], items["codes"]))
meta = items.set_index("var")[["split_60_20_20", "scale_type", "sensitivity_level", "topic"]].rename(
    columns={"split_60_20_20": "split"})
H = {}
rows, info = [], {}
for model, paths in SOURCES.items():
    recs = read(paths)
    info[model] = {"files": [p.name for p in paths], "records": len(recs)}
    cnt = collections.defaultdict(collections.Counter)
    ntot, nref = collections.Counter(), collections.Counter()
    for (pid, _), (valid, parsed, inv) in recs.items():
        p = probes[pid]
        key = (p["cell_id"], p["var"])
        ntot[key] += 1
        nref[key] += inv == "refusal"
        if valid:
            cnt[key][parsed] += 1
    for (persona, var), nt in sorted(ntot.items()):
        codes = item_codes[var]
        if (persona, var) not in H:
            H[(persona, var)] = human(persona, var, codes)
        hp, neff = H[(persona, var)]
        nv = sum(cnt[(persona, var)].values())
        for k, c in enumerate(codes):
            rows.append(dict(model=model, level=LEVEL[persona], persona=persona, var=var, code=c, opt_idx=k,
                             n_options=len(codes), count=cnt[(persona, var)][c], n_valid=nv, n_total=nt,
                             n_refusal=nref[(persona, var)], human_p=hp[k], human_neff=neff))

df = pd.DataFrame(rows).join(meta, on="var")
df = df.astype({"count": "int32", "n_valid": "int32", "n_total": "int32", "n_refusal": "int32"})
df.to_csv(HERE / "kosmos_long.csv.gz", index=False, float_format="%.6g")
per = df.drop_duplicates(["model", "persona", "var"]).groupby("model")["n_total"].agg(["min", "median", "max"])
info["k_per_prompt"] = per.to_dict("index")
info["built_at"] = datetime.datetime.now().isoformat(timespec="seconds")
(HERE / "kosmos_build.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(df.shape, "\n", per)
