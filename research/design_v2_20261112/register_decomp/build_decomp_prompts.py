"""인칭·화계 요소 분해 프롬프트 생성 — 실행 (KOSMOS 루트):
  uv run -q --no-project --python 3.12 --with pandas python research/design_v2_20261112/register_decomp/build_decomp_prompts.py

register_v1 확정본(검수 끝난 문장)을 부품 단위로 다시 조립한다. 새 한국어 변환은 하지 않는다.

한국 성인(L0) 프롬프트의 부품:
  페르소나 문장  "당신은 한국 성인입니다." / "이 사람은 한국 성인입니다."
  문항 호칭      귀하(원문) / 이 사람 (register_v1의 C_with_si: 주체 높임 -시- 유지)
  지시문         "번호로만 답하십시오." / "이 사람이 고를 번호로만 답하십시오."
  주체 높임      -시- 유지 / 제거 (register_v1의 이 사람·하십시오체는 제거한 판)
화계는 모두 하십시오체다. system도 원문 그대로다.

기존 조건과의 관계:
  원문(당신·하십시오체)      = 페르소나 당신, 호칭 귀하, 지시문 원문, -시- 유지
  이 사람·하십시오체(기존)   = 페르소나 이 사람, 호칭 이 사람, 지시문 이 사람, -시- 제거
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "register_v1" / "out"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

PERSONA = {"당신": "당신은 한국 성인입니다.", "이 사람": "이 사람은 한국 성인입니다."}
INSTR = {"원문": "번호로만 답하십시오.", "이 사람": "이 사람이 고를 번호로만 답하십시오."}

# cond: (이름, 페르소나, 문항 호칭, 지시문)   호칭 값: 귀하 / 이 사람 / 당신 (모두 -시- 유지)
CONDS = {
    "persona":         ("페르소나 문장만 '이 사람'",            "이 사람", "귀하",   "원문"),
    "address":         ("문항 호칭만 '이 사람'",                "당신",   "이 사람", "원문"),
    "instr":           ("지시문만 '이 사람이 고를'",            "당신",   "귀하",   "이 사람"),
    "persona_address": ("페르소나 문장 + 문항 호칭",            "이 사람", "이 사람", "원문"),
    "persona_instr":   ("페르소나 문장 + 지시문",               "이 사람", "귀하",   "이 사람"),
    "address_instr":   ("문항 호칭 + 지시문",                   "당신",   "이 사람", "이 사람"),
    "all_keep_si":     ("세 요소 모두 '이 사람', 주체 높임 유지", "이 사람", "이 사람", "이 사람"),
    "dangsin_address": ("문항 호칭만 귀하→당신 (하십시오체)",     "당신",   "당신",   "원문"),
}

PAT = re.compile(r"^(?P<persona>[^\n]+)\n\n문항: (?P<stem>.*?)\n\n선택지:\n(?P<opts>.*?)\n\n(?P<instr>[^\n]+)$", re.S)

reg = pd.read_csv(V1 / "prompts_register_v1.csv", encoding="utf-8-sig")
leads = pd.read_csv(V1 / "lead_table.csv", encoding="utf-8-sig").fillna("")
L0 = reg[reg.level == "L0"]
A = L0[L0.cond == "A"].set_index("var")
C = L0[L0.cond == "C"].set_index("var")
var2lead = {v: r for _, r in leads.iterrows() for v in r["vars"].split()}


def parts(user):
    m = PAT.match(user)
    assert m, user[:80]
    return m.groupdict()


def assemble(persona, stem, opts, instr):
    return f"{persona}\n\n문항: {stem}\n\n선택지:\n{opts}\n\n{instr}"


rows, flags = [], []
for v in A.index:
    a, c = parts(A.loc[v, "user"]), parts(C.loc[v, "user"])
    assert a["opts"] == c["opts"], v
    lt = var2lead[v]
    # 원문 stem = A lead + 나머지, 이 사람 stem(-시- 제거) = C lead + 나머지. 나머지(진술문)는 각자 것을 쓴다.
    assert a["stem"].startswith(lt["A"]), v
    assert c["stem"].startswith(lt["C"]), v
    rest_a, rest_c = a["stem"][len(lt["A"]):], c["stem"][len(lt["C"]):]
    stem = {"귀하": a["stem"], "이 사람": lt["C_with_si"] + rest_c}
    # 이 사람·당신은 둘 다 받침으로 끝나 조사가 같다 → 단순 치환이 조사까지 맞는다.
    stem["당신"] = stem["이 사람"].replace("이 사람", "당신")
    # 회귀 점검: 부품으로 기존 두 조건을 똑같이 다시 만들 수 있어야 한다.
    assert assemble(PERSONA["당신"], stem["귀하"], a["opts"], INSTR["원문"]) == A.loc[v, "user"], v
    assert assemble(PERSONA["이 사람"], lt["C"] + rest_c, a["opts"], INSTR["이 사람"]) == C.loc[v, "user"], v
    if lt["C_with_si"] == lt["C"]:
        flags.append(dict(var=v, flag="주체 높임 없는 문항 (all_keep_si = 기존 이 사람·하십시오체와 같은 문장)"))
    if "귀하" not in a["stem"] and "귀댁" not in a["stem"]:
        flags.append(dict(var=v, flag="원문에 2인칭 없음 (register_v1 보정으로 '이 사람'을 넣은 문항)"))
    for cond, (name, p, adr, ins) in CONDS.items():
        rows.append(dict(probe_id=f"L0__{v}__{cond}", phase=1, level="L0", cell_id="L0", var=v, codes=A.loc[v, "codes"],
                         cond=cond, cond_name=name, persona=p, address=adr, instr=ins,
                         system=A.loc[v, "system"], user=assemble(PERSONA[p], stem[adr], a["opts"], INSTR[ins])))

df = pd.DataFrame(rows)
df.to_csv(OUT / "prompts_decomp.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(flags).to_csv(OUT / "flags.csv", index=False, encoding="utf-8-sig")
sha = hashlib.sha256((OUT / "prompts_decomp.csv").read_bytes()).hexdigest()

# 검수용 예시: 문항 3개 × 조건 전부(기존 두 조건 포함)
ex = ["ACPTCULT", "EMPLY", "AUTHORT4"]
with open(OUT / "examples.md", "w", encoding="utf-8") as f:
    f.write("# 요소 분해 프롬프트 예시 (자동 생성)\n\n")
    for v in ex:
        if v not in A.index:
            continue
        f.write(f"## {v}\n\n### 원문 (기존)\n```\n{A.loc[v, 'user']}\n```\n\n### 이 사람·하십시오체 (기존)\n```\n{C.loc[v, 'user']}\n```\n\n")
        for _, r in df[df["var"] == v].iterrows():
            f.write(f"### {r.cond} — {r.cond_name}\n```\n{r.user}\n```\n\n")

summary = {"prompts_sha256": sha, "n_rows": len(df), "n_items": int(df["var"].nunique()),
           "conds": {k: v[0] for k, v in CONDS.items()}, "flags": pd.DataFrame(flags)["flag"].value_counts().to_dict() if flags else {}}
json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
