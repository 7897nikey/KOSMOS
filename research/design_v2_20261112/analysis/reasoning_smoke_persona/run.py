"""DeepSeek 추론 low 스모크(Eco, 프롬프트당 1회)의 페르소나 반응성 탐색.

지표 (L1 응답 a, 문항 v, 셀 c, 사람 분포를 Jeffreys 평활):
  LLR(a) = log p_c(a) - log p_pop(a)   p_c: 그 셀 사람 분포, p_pop: 모집단(L0) 사람 분포
  양수면 답이 모집단보다 그 셀 사람에게 더 전형적이다 → 평균 LLR = 페르소나 반응성.
  null: 같은 문항 안에서 셀 라벨을 섞어 다시 계산(순열 2,000회).
  적합도: log p_c(a) 평균 (높을수록 셀 사람이 그 답을 많이 함).
비교 기준(추론 끔): 데이터셋의 deepseek(= Ultrafast, 추론 끔, k=100)·exaone(k≈98)·haiku 분포에서
  같은 (셀, 문항)의 1회 뽑기 기댓값 Σ q_c(a)·지표(a). Eco≠Ultrafast라 엄밀한 짝 비교는 아니다.
test 문항은 제외한다(사전등록: test는 k=100 확정 후 1회만).
"""
import json, pathlib, datetime, hashlib, subprocess
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[4]; OUT = pathlib.Path(__file__).parent
DATA = ROOT/"research/design_v2_20261112/dataset/kosmos_long.csv.gz"
SMOKE = ROOT/"raw/v2/deepseek_eco_low_p1_smoke.jsonl"
rng = np.random.default_rng(20261118)

df = pd.read_csv(DATA)
df["code"] = df["code"].astype(str)
df = df[df["split"] != "test"]
h = df[df["model"] == "deepseek"][["level", "persona", "var", "code", "n_options", "human_p", "human_neff"]].copy()
h["hp"] = (h["human_p"]*h["human_neff"] + 0.5) / (h["human_neff"] + 0.5*h["n_options"])
pop = h[h["level"] == "L0"].set_index(["var", "code"])["hp"]
cell = h[h["level"] == "L1"].set_index(["persona", "var", "code"])["hp"]

sm = [json.loads(l) for l in open(SMOKE)]
sm = [r for r in sm if r["level"] == "L1" and r["valid"] and (r["var"], str(r["parsed"])) in pop.index
      and (r["cell_id"], r["var"], str(r["parsed"])) in cell.index]
cells_by_var = h[h["level"] == "L1"].groupby("var")["persona"].unique().to_dict()

def llr(c, v, a): return np.log(cell[(c, v, a)]) - np.log(pop[(v, a)])
def fit(c, v, a): return np.log(cell[(c, v, a)])

obs = np.array([llr(r["cell_id"], r["var"], str(r["parsed"])) for r in sm])
fitv = np.array([fit(r["cell_id"], r["var"], str(r["parsed"])) for r in sm])
# 순열 null: 같은 문항 안 다른 셀로 라벨 교체
null = []
for _ in range(2000):
    null.append(np.mean([llr(rng.choice(cells_by_var[r["var"]]), r["var"], str(r["parsed"])) for r in sm]))
null = np.array(null)
boot = np.array([obs[rng.integers(0, len(obs), len(obs))].mean() for _ in range(2000)])
bootf = np.array([fitv[rng.integers(0, len(fitv), len(fitv))].mean() for _ in range(2000)])

# 추론 끔 기준: 같은 (셀, 문항)에서 원비율 q로 1회 뽑을 때의 기댓값, 그리고 같은 문항 다른 셀 라벨일 때의 기댓값
keys = {(r["cell_id"], r["var"]) for r in sm}
base = {}
for m in ["deepseek", "exaone", "haiku"]:
    d = df[(df["model"] == m) & (df["level"] == "L1")].copy()
    d["q"] = d["count"] / d["n_valid"].clip(lower=1)
    e_llr, e_fit, e_null = [], [], []
    for (c, v), g in d.groupby(["persona", "var"]):
        if (c, v) not in keys: continue
        q = g.set_index("code")["q"]; q.index = q.index.astype(str)
        e_llr.append(sum(q[a]*llr(c, v, a) for a in q.index))
        e_fit.append(sum(q[a]*fit(c, v, a) for a in q.index))
        e_null.append(np.mean([sum(q[a]*llr(c2, v, a) for a in q.index) for c2 in cells_by_var[v]]))
    base[m] = {"n": len(e_llr), "llr": float(np.mean(e_llr)), "llr_null": float(np.mean(e_null)),
               "llr_minus_null": float(np.mean(e_llr) - np.mean(e_null)), "fit": float(np.mean(e_fit))}

res = {"smoke_deepseek_eco_low": {
        "n": len(obs), "llr": float(obs.mean()), "llr_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "llr_null_mean": float(null.mean()), "llr_minus_null": float(obs.mean() - null.mean()),
        "perm_p_one_sided": float((null >= obs.mean()).mean()),
        "fit": float(fitv.mean()), "fit_ci95": [float(x) for x in np.percentile(bootf, [2.5, 97.5])]},
       "reasoning_off_expected_single_draw": base}
# 추론 본문이 페르소나(성별·나이)를 언급하는 비율
rows = [json.loads(l) for l in open(SMOKE)]
import re
pat = re.compile(r"(male|female|man|woman|\b\d{2}\s*-?\s*(year|yo|s\b)|age|older|younger|elderly|young|남성|여성|세)", re.I)
for lv in ["c0", "L0", "L1"]:
    rr = [r for r in rows if r["level"] == lv]
    res.setdefault("reasoning_mentions_persona", {})[lv] = f"{sum(bool(pat.search(r.get('reasoning_text',''))) for r in rr)}/{len(rr)}"
res["_meta"] = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                "git_commit": subprocess.run(["git","-C",str(ROOT),"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
                "smoke_sha256_16": hashlib.sha256(SMOKE.read_bytes()).hexdigest()[:16],
                "data_sha256_16": hashlib.sha256(DATA.read_bytes()).hexdigest()[:16], "include_test": False, "paid_calls": 0}
json.dump(res, open(OUT/"run.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
