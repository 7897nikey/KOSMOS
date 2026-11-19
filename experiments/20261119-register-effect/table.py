"""20261119-register-effect 보조 — 해석 없는 전체 결과표.
실행 (KOSMOS 루트): python experiments/20261119-register-effect/table.py  (levels.py를 먼저 돌린다)

입력: out/levels.csv(인칭·화계 18조건 지표), 원응답(응답·무효 건수),
      experiments/20261118-reasoning-effect/out/diffs.csv(추론 효과 실험의 성별×연령 10셀 지표).
출력: out/results_table.csv, out/results_table_cells.csv, RESULTS_TABLE.md, out/table_run.json
"""
import collections
import csv
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DESIGN = ROOT / "research/design_v2_20261112"
out = HERE / "out"

lev = pd.read_csv(out / "levels.csv", encoding="utf-8-sig")

# 응답·무효 건수: run.py와 같은 규칙(표본 0~49, 중복·api_error 제외)
meta = {}
for r in csv.DictReader((DESIGN / "prompts_v2.csv").open(encoding="utf-8-sig")):
    if r["phase"] == "1" and r["level"] in ("c0", "L0"):
        meta[r["probe_id"]] = (r["level"], "A")
for r in csv.DictReader((DESIGN / "register_v1/out/prompts_register_v1.csv").open(encoding="utf-8-sig")):
    if r["reuse_existing"] == "False":
        meta[r["probe_id"]] = (r["level"], r["cond"])
RAW = {"끔": ["raw/v2/deepseek_eco_p1.jsonl", "raw/v2/deepseek_eco_register_p1.jsonl"],
       "low": ["raw/v2/deepseek_eco_low_p1.jsonl", "raw/v2/deepseek_eco_low_register_p1.jsonl"],
       "haiku": ["raw/v2/haiku_p1.jsonl", "raw/v2/haiku_register_p1.jsonl"]}
n, bad = collections.Counter(), collections.Counter()
for s, files in RAW.items():
    seen = set()
    for f in files:
        for line in open(ROOT / f, encoding="utf-8"):
            r = json.loads(line)
            key = (r["probe_id"], r["sample_idx"])
            if r.get("invalid_type") == "api_error" or r["sample_idx"] >= 50 or key in seen or r["probe_id"] not in meta:
                continue
            seen.add(key)
            k = (s, *meta[r["probe_id"]])
            n[k] += 1
            bad[k] += not r["valid"]
lev["n_responses"] = [n[(s, l, c)] for s, l, c in zip(lev.setting, lev.level, lev.cond)]
lev["n_invalid"] = [bad[(s, l, c)] for s, l, c in zip(lev.setting, lev.level, lev.cond)]

COLS = [("model", "모델"), ("persona", "페르소나"), ("style", "문체"),
        ("tvd_beta1", "보정 전 TVD (197문항)"), ("tvd_beta1_val", "보정 전 TVD (val 49)"),
        ("beta", "β"), ("tvd_calibrated_cal148", "보정 후 TVD (cal 148)"), ("tvd_calibrated_val", "보정 후 TVD (val 49)"),
        ("mode_agree", "최빈값 일치율 (197)"), ("pair_agree", "보기 순서쌍 일치율 (197)"),
        ("n_responses", "응답 수"), ("n_invalid", "무효 수")]
T = lev[[c for c, _ in COLS]].rename(columns=dict(COLS))
T.to_csv(out / "results_table.csv", index=False, encoding="utf-8-sig")

d = pd.read_csv(ROOT / "experiments/20261118-reasoning-effect/out/diffs.csv")
d = d[d["용도"] == "판정"].set_index("지표")
CELL = [("D1 보정 TVD L1", "보정 후 TVD, 성별×연령 10셀 (val 49)"), ("D3 셀 간 AUC", "집단 간 순서쌍 일치율 (197 중 서열·이진)"),
        ("D4 보기 순서 AUC", "셀 안 보기 순서쌍 일치율 (197)"), ("D5 최빈값 일치 L1", "최빈값 일치율, 성별×연령 10셀 평균 (197)")]
U = pd.DataFrame([{"지표": lab, "DeepSeek 추론 끔": d.loc[k, "끔"], "DeepSeek 추론 low": d.loc[k, "low"]} for k, lab in CELL])
U.to_csv(out / "results_table_cells.csv", index=False, encoding="utf-8-sig")


def md(df, fmt):
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" if df[c].dtype == object else "---:" for c in df.columns) + "|\n"
    body = "".join("| " + " | ".join(fmt(c, v) for c, v in row.items()) + " |\n" for _, row in df.iterrows())
    return head + body


def f(c, v):
    if isinstance(v, str):
        return v
    if c in ("응답 수", "무효 수"):
        return f"{int(v):,}"
    if c == "β":
        return f"{v:.3f}"
    return "—" if pd.isna(v) else f"{v:.3f}"


text = f"""# 인칭·화계 실험 결과표

해석 없이 지표 값만 담은 표다. 생성: `table.py` (입력 `out/levels.csv`, 원응답, 추론 효과 실험 `out/diffs.csv`). 숫자는 소수 셋째 자리 반올림, 원값은 `out/results_table.csv`·`out/results_table_cells.csv`.

- 모델: DeepSeek V4.1 Flash Eco(추론 끔·추론 low), Claude Haiku 5.5(추론 끔). 프롬프트당 50회 표집.
- 문항: 246문항 중 test 49문항 제외. calibration 148 + validation 49 = 197.
- 보정 전 TVD: 보기 개수에 1을 더해 정규화한 모델 분포와 KGSS 2023 가중 분포의 TVD, 문항 평균. 낮을수록 사람과 가깝다.
- 보정 후 TVD: p ∝ (개수+1)^β. β는 0.05~1.5(0.005 간격)에서 calibration 148문항 평균 TVD가 가장 작은 값을 모델·페르소나·문체마다 고르고, validation 49문항에서 잰다.
- 최빈값 일치율: 모델 최빈 보기가 사람 최빈 보기와 같은 문항 비율(동점 규칙 미적용).
- 보기 순서쌍 일치율: 사람 비율이 다른 보기 쌍(1,623쌍)에서 모델 비율의 대소가 같은 비율. 같으면 1, 모델 동률 0.5. 무작위 0.5.
- 무효 수: 숫자 하나로 파싱되지 않은 응답(거부·잘림 포함).

## 페르소나 × 문체 (18조건)

{md(T, f)}
## 성별×연령 10셀 (추론 효과 실험, 원문 문체, DeepSeek만)

{md(U, f)}"""
(HERE / "RESULTS_TABLE.md").write_text(text, encoding="utf-8")
print(text)

git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
json.dump({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git("rev-parse", "--short", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
           "inputs": {"levels.csv": hashlib.sha256((out / "levels.csv").read_bytes()).hexdigest()[:16]},
           "include_test": False}, open(out / "table_run.json", "w"), ensure_ascii=False, indent=1)
