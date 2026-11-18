"""2단계 실험 계획의 호출 수·비용·시간 추정 (유료 호출 없음).
토큰: 같은 모델의 실측 원응답에서 단계별 평균 (DeepSeek 끔 = Ultrafast 1단계, DeepSeek low = Eco low 스모크, Haiku 끔 = 1단계 배치).
L2는 L1 토큰을 쓴다. 3인칭·해라체 프롬프트 길이 차이(약 +5토큰)는 무시한다.
속도: DeepSeek Eco = 11/18 22:3x 동시 실행 실측(끔 11건/s, low 6건/s, 같은 키 분당 한도 공유),
      Haiku Batch = 11/11 1단계 295,200건 3시간 37분, K-EXAONE = 키 1개 초당 1회 한도, 실측 73,800건 24~41시간(0.5~0.85건/s).
"""
import json, collections, pathlib, csv
ROOT = pathlib.Path(__file__).resolve().parents[3]
def per_level(f, pi, po):
    s = collections.defaultdict(lambda: [0, 0, 0])
    for l in open(ROOT / f):
        r = json.loads(l)
        if r.get("invalid_type") == "api_error" or r.get("input_tokens") is None: continue
        a = s[r["level"]]; a[0] += 1; a[1] += r["input_tokens"]; a[2] += r["output_tokens"]
    return {lv: (a[1]/a[0]*pi + a[2]/a[0]*po)/1e6 for lv, a in s.items()}
COST = {"deepseek_off": per_level("raw/v2/deepseek_ultrafast_p1.jsonl", .03, .12),
        "deepseek_low": per_level("raw/v2/deepseek_eco_low_p1_smoke.jsonl", .03, .12),
        "haiku_off": per_level("raw/v2/haiku_p1.jsonl", .05, .25)}
for c in COST.values(): c["L2"] = c["L1"]
COST["exaone_low"] = {lv: 0.0 for lv in ["c0", "L0", "L1", "L2"]}
RATE = {"deepseek_off": (11, 11), "deepseek_low": (6, 6), "haiku_off": (22.7, 22.7), "exaone_low": (0.5 * 4, 0.85 * 4)}  # 건/s (느림, 빠름); EXAONE은 팀원 4명 키
N = {"c0": 246, "L0": 246, "L1": 2460, "L2": 2952}
K = 50
EXPS = [
    ("E0", "추론 효과 (기본 문체)", ["c0", "L0", "L1"], 1, ["deepseek_off", "deepseek_low"], "진행 중"),
    # 2026-11-18 사용자 결정: 인칭×화계는 비추론 축과 추론 축을 모두 돌린다. 같은 조건(k=50, 같은 프롬프트)으로 Haiku도 비추론 축에 넣는다.
    # Haiku는 추론 축에서 뺀다(effort low 1/300, medium 17/300만 생각해 추론 조건이 성립하지 않음). K-EXAONE은 추론을 끌 수 없어 추론 축에만 있다.
    # 2026-11-19 팀 결정(Slack #26-2_nlp)으로 위 E1a/E1b(L0+L1 × 문체 3개)를 아래 E1로 바꾼다:
    # 페르소나는 c0(화계 B 1개)·L0(B·C·D 3개)만, 프롬프트는 register_v1 확정본 신규 984개. EXAONE은 제외(유료 모델 위주).
    # A 조건은 기존 데이터(E0, Haiku 1단계 표본 0~49)를 쓴다. 표기: "c0:1"은 c0를 문체 1개만.
    ("E1", "인칭×화계 (register_v1 신규 984개)", ["c0:1", "L0:3"], None, ["deepseek_off", "deepseek_low", "haiku_off"], "필수"),
    # L2는 DESIGN.md대로 1단계 결과를 보고 정한다. 1단계에서 L0→L1이 세 모델 모두 나아지지 않아 보류 (2026-11-18)
    ("E2", "L2 12셀 (기본 문체)", ["L2"], 1, ["deepseek_off", "haiku_off"], "보류"),  # EXAONE 제외 (2026-11-19)
    ("E2+", "L2 × 추론", ["L2"], 1, ["deepseek_low"], "보류"),
]
rows = []
for eid, name, levels, styles, arms, status in EXPS:
    lv_st = [(l.split(":")[0], int(l.split(":")[1])) if ":" in l else (l, styles) for l in levels]
    for arm in arms:
        calls = sum(N[l] * st for l, st in lv_st) * K
        usd = sum(N[l] * st * K * COST[arm][l] for l, st in lv_st)
        slow, fast = RATE[arm]
        rows.append(dict(exp=eid, name=name, arm=arm, status=status, levels="+".join(levels), styles=styles or "-", k=K,
                         calls=calls, usd=round(usd, 2), hours_fast=round(calls / fast / 3600, 1), hours_slow=round(calls / slow / 3600, 1)))
with open(pathlib.Path(__file__).parent / "cost_plan.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
tot = collections.defaultdict(lambda: [0, 0.0])
for r in rows:
    tot[r["status"]][0] += r["calls"]; tot[r["status"]][1] += r["usd"]
print({k: (v[0], round(v[1], 2)) for k, v in tot.items()})
print({a: {l: round(v * 1e6, 2) for l, v in c.items()} for a, c in COST.items()}, "(USD per 1M calls)")
