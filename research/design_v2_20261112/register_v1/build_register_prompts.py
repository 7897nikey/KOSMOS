"""인칭·화계 프롬프트 변환 초안 — 실행: PYTHONUTF8=1 python research/design_v2_20261112/register_v1/build_register_prompts.py

prompts_v2.csv(설계 고정 파일, 읽기만 한다)의 c0·L0 프롬프트를 인칭(당신/이 사람) × 화계(하십시오체/해라체)
조건으로 바꾼다. c0는 A·B 2조건, L0는 A·B·C·D 4조건이다.

  A 당신·하십시오체  = 원문 그대로 (기존 수집 데이터와 같아야 한다)
  B 당신·해라체
  C 이 사람·하십시오체
  D 이 사람·해라체

바꾸는 곳: system, 페르소나 문장(L0), 문항 앞부분(“...” 앞, 이하 '도입부'), 마지막 지시문.
바꾸지 않는 곳: “...” 뒤 진술문, 선택지(존댓말이 없음을 점검한다).

순서: (1) 규칙 변환 → (2) corrections.csv의 사람/LLM 보정으로 덮어쓰기 → (3) 자동 점검 → (4) 검수용 대조표.
"""
import csv
import difflib
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "prompts_v2.csv"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

CONDS = {"A": ("당신", "하십시오체"), "B": ("당신", "해라체"), "C": ("이 사람", "하십시오체"), "D": ("이 사람", "해라체")}
LEVEL_CONDS = {"c0": ["A", "B"], "L0": ["A", "B", "C", "D"]}
SYSTEM = {"하십시오체": "보기 번호 숫자 하나만 출력하고 다른 말은 하지 마십시오.",
          "해라체": "보기 번호 숫자 하나만 출력하고 다른 말은 하지 마라."}
PERSONA = {"A": "당신은 한국 성인입니다.", "B": "당신은 한국 성인이다.",
           "C": "이 사람은 한국 성인입니다.", "D": "이 사람은 한국 성인이다."}
INSTR = {"A": "번호로만 답하십시오.", "B": "번호로만 답해라.",
         "C": "이 사람이 고를 번호로만 답하십시오.", "D": "이 사람이 고를 번호로만 답해라."}

# ── (1) 규칙 ──────────────────────────────────────────
# 인칭: 귀하·귀댁 → 당신 / 이 사람. 귀하는 모음, 당신·사람은 자음으로 끝나 조사가 바뀐다.
PRON = [("귀하께서는", "{P}은"), ("귀하께서", "{P}이"), ("귀하에게는", "{P}에게는"), ("귀하에게", "{P}에게"),
        ("귀하는", "{P}은"), ("귀하가", "{P}이"), ("귀하를", "{P}을"), ("귀하와", "{P}과"), ("귀하의", "{P}의"),
        ("귀하 ", "{P} "), ("귀댁의", "{P} 가정의"), ("귀댁에서", "{P}의 집에서")]
# 화계: 하십시오체(주체 높임 -시- 포함) → 해라체. 긴 패턴부터.
ENDINGS = [
    ("말씀해 주시기 바랍니다", "말해 주기 바란다"), ("말씀해주시기 바랍니다", "말해 주기 바란다"),
    ("답하여 주시기 바랍니다", "답해 주기 바란다"), ("말씀해 주십시오", "말해라"), ("말씀해주십시오", "말해라"),
    ("말씀하여 주십시오", "말해라"), ("말씀해 주시고", "말하고"), ("답하여 주십시오", "답해라"),
    ("골라 주십시오", "골라라"), ("염두에 두십시오", "염두에 두어라"),
    ("주시겠습니까", "주겠는가"), ("있으셨습니까", "있었는가"), ("있으십니까", "있는가"), ("계십니까", "있는가"),
    ("이십니까", "인가"), ("하십니까", "하는가"), ("느끼십니까", "느끼는가"), ("가지십니까", "가지는가"),
    ("치르십니까", "치르는가"), ("입니까", "인가"), ("됩니까", "되는가"), ("있습니까", "있는가"),
    ("싶습니까", "싶은가"), ("있었습니까", "있었는가"),
    ("하시는지", "하는지"), ("시는지", "는지"), ("하실 때", "할 때"), ("보시기에", "보기에"),
    ("여쭤보겠습니다", "묻겠다"), ("여쭈어 보겠습니다", "묻겠다"), ("있습니다", "있다"), ("입니다", "이다"),
    ("말합니다", "말한다"), ("보도합니다", "보도한다"),
]


# 3인칭 하십시오체(C)에서는 주체 높임 -시-를 뺀다(팀 결정 2026-11-19, 쟁점 1).
# 하십시오체 어미(-ㅂ니까/-습니까)는 듣는 사람(모델)을, -시-는 주어(이 사람)를 높인다.
# -시-를 빼야 C와 D의 차이가 '모델에게 하는 말투'만 남는다. 듣는 사람에게 하는 요청
# (말씀해 주십시오, 주시기 바랍니다, 두십시오 등)은 그대로 둔다.
SUBJ_HON = [
    ("가셨습니까", "갔습니까"), ("가지십니까", "가집니까"), ("계십니까", "있습니까"),
    ("구성원이십니까", "구성원입니까"), ("느끼셨습니까", "느꼈습니까"), ("느끼십니까", "느낍니까"),
    ("받으셨습니까", "받았습니까"), ("받으신", "받은"), ("보시기에", "보기에"),
    ("응해주실 의향이 있으십니까", "응할 의향이 있습니까"), ("있으셨습니까", "있었습니까"),
    ("있으십니까", "있습니까"), ("졸업하셨습니까", "졸업했습니까"), ("주시겠습니까", "주겠습니까"),
    ("참여하셨습니까", "참여했습니까"), ("참여하실", "참여할"), ("치르십니까", "치릅니까"),
    ("투표하셨습니까", "투표했습니까"), ("투표하실", "투표할"),
    ("여기시는지", "여기는지"), ("하시는지", "하는지"), ("하십니까", "합니까"),
]
SUBJ_HON_LEFT = re.compile(r"십니까|셨습|시는지|보시기|하실|으신|이십|계십|주시겠|주실")


def drop_subject_honorific(text):
    for a, b in SUBJ_HON:
        text = text.replace(a, b)
    return text


def apply_pron(text, p):
    for a, b in PRON:
        text = text.replace(a, b.format(P=p))
    return text


def apply_endings(text):
    for a, b in ENDINGS:
        text = text.replace(a, b)
    return text


def rule(lead_a, cond):
    p, reg = CONDS[cond]
    t = apply_pron(lead_a, p)
    if reg == "해라체":
        t = apply_endings(t)
    return t


# ── 원문 분해 ────────────────────────────────────────
src = pd.read_csv(SRC, encoding="utf-8")
src = src[(src.phase == 1) & (src.level.isin(["c0", "L0"]))].copy()
PAT = re.compile(r"^(?:(?P<persona>[^\n]+)\n\n)?문항: (?P<stem>.*?)\n\n선택지:\n(?P<opts>.*?)\n\n(?P<instr>[^\n]+)$", re.S)
rows = []
for r in src.itertuples():
    m = PAT.match(r.user)
    assert m, r.probe_id
    stem = m["stem"]
    lead, sep, stmt = stem.partition("...")
    rows.append(dict(probe_id=r.probe_id, level=r.level, var=r.var, codes=r.codes, system=r.system, user=r.user,
                     persona=m["persona"], lead=lead, sep=sep, stmt=stmt, opts=m["opts"], instr=m["instr"]))
P = pd.DataFrame(rows)
assert (P.loc[P.level == "L0", "persona"] == PERSONA["A"]).all()
assert P.loc[P.level == "c0", "persona"].isna().all()
assert (P.instr == INSTR["A"]).all() and (P.system == SYSTEM["하십시오체"]).all()

leads = P.drop_duplicates("lead")[["var", "lead"]].rename(columns={"var": "lead_id", "lead": "A"}).reset_index(drop=True)
leads["n_items"] = leads["A"].map(P.drop_duplicates("var").groupby("lead").size())
leads["vars"] = leads["A"].map(P.drop_duplicates("var").groupby("lead")["var"].apply(lambda s: " ".join(s)))
for c in "BCD":
    leads[f"{c}_rule"] = [rule(a, c) for a in leads["A"]]

# ── (2) 보정 ─────────────────────────────────────────
corr_path = HERE / "corrections.csv"
corr = pd.read_csv(corr_path, encoding="utf-8", dtype=str).fillna("") if corr_path.exists() else \
    pd.DataFrame(columns=["lead_id", "field", "cond", "text", "note"])
stmt_corr = {(x.lead_id, x.cond): x.text for x in corr[corr.field == "stmt"].itertuples()}
for (v, _c) in stmt_corr:
    assert v in set(P["var"]), f"corrections.csv의 문항 없음: {v}"
corr = corr[corr.field == "lead"]
for c in "BCD":
    leads[c] = leads[f"{c}_rule"]
    leads[f"{c}_src"] = "규칙"
    leads[f"{c}_note"] = ""
for x in corr.itertuples():
    i = leads.index[leads.lead_id == x.lead_id]
    assert len(i) == 1, f"corrections.csv의 lead_id 없음: {x.lead_id}"
    leads.loc[i, x.cond] = x.text
    leads.loc[i, f"{x.cond}_src"] = "보정"
    leads.loc[i, f"{x.cond}_note"] = x.note
# C: 규칙·보정 결과 모두에 주체 높임 제거를 마지막으로 적용한다.
leads["C_with_si"] = leads["C"]
leads["C"] = leads["C"].map(drop_subject_honorific)

# ── (3) 자동 점검 ────────────────────────────────────
HON = re.compile(r"귀하|귀댁|께서|니까|십시오|습니다|[가-힣]ㅂ니다|입니다|합니다|됩니다|십니|시는|시기|드립|세요")
NORM_PRON = re.compile(r"귀하께서는|귀하께서|귀하에게는|귀하에게|귀하는|귀하가|귀하를|귀하와|귀하의|귀하|귀댁의|귀댁에서|귀댁|"
                       r"당신은|당신이|당신을|당신과|당신의|당신에게는|당신에게|당신|이 사람은|이 사람이|이 사람을|이 사람과|"
                       r"이 사람의|이 사람에게는|이 사람에게|이 사람")
NOUNS = re.compile(r"[가-힣A-Za-z0-9]{2,}")


# 화계를 바꾸면 당연히 바뀌는 말(요청·어미 동사)은 내용어 비교에서 뺀다.
FUNC = {"말씀", "바랍", "바란", "주십", "주시", "주기", "말해", "말하", "답하", "답해", "여쭤", "여쭈", "묻겠"}


def content_words(t):
    t = NORM_PRON.sub(" ", t)
    return {w[:2] for w in NOUNS.findall(t)} - FUNC


def checks(row, c):
    t, a = row[c], row["A"]
    pron, reg = CONDS[c]
    flags = []
    if reg == "해라체" and HON.search(t):
        flags.append(f"존댓말 남음({HON.search(t).group()})")
    if reg == "하십시오체" and not re.search(r"니까|십시오|니다|바랍니다", t):
        flags.append("하십시오체 어미 없음")
    if pron == "이 사람" and ("귀하" in t or "귀댁" in t or "당신" in t):
        flags.append("2인칭 남음")
    if pron == "이 사람" and "이 사람" not in t:
        flags.append("'이 사람' 없음")
    if pron == "당신" and c != "A" and ("귀하" in t or "귀댁" in t):
        flags.append("귀하 남음")
    if c == "C" and SUBJ_HON_LEFT.search(t):
        flags.append(f"주체 높임 남음({SUBJ_HON_LEFT.search(t).group()})")
    lost = content_words(a) - content_words(t)
    if len(lost) > 2:
        flags.append("내용어 변화 " + ",".join(sorted(lost)[:6]))
    sim = difflib.SequenceMatcher(None, NORM_PRON.sub("", a), NORM_PRON.sub("", t)).ratio()
    if sim < 0.6:
        flags.append(f"유사도 {sim:.2f}")
    return "; ".join(flags), round(sim, 3)


for c in "BCD":
    res = [checks(r, c) for _, r in leads.iterrows()]
    leads[f"{c}_check"] = [x[0] for x in res]
    leads[f"{c}_sim"] = [x[1] for x in res]

# 진술문·선택지에 존댓말이 없는지 (조건마다 그대로 쓰기 때문)
opt_hon = P[P.opts.str.contains(r"귀하|니까|십시오|습니다|입니다|세요")]
stmt_hon = P[P.stmt.str.contains(r"귀하|니까|십시오|습니다|입니다|세요")]

# ── 프롬프트 조립 ────────────────────────────────────
lead_map = leads.set_index("A")
out_rows = []
for r in P.itertuples():
    for c in LEVEL_CONDS[r.level]:
        pron, reg = CONDS[c]
        lead = r.lead if c == "A" else lead_map.loc[r.lead, c]
        stmt = (" " + stmt_corr[(r.var, c)]) if (r.var, c) in stmt_corr else r.stmt
        stem = lead + r.sep + stmt
        parts = ([PERSONA[c], ""] if r.level == "L0" else []) + [f"문항: {stem}", "", "선택지:", r.opts, "", INSTR[c]]
        user = "\n".join(parts)
        system = SYSTEM[reg]
        if c == "A":
            assert user == r.user and system == r.system, f"A가 원문과 다름: {r.probe_id}"
        out_rows.append(dict(probe_id=f"{r.level}__{r.var}__{c}", level=r.level, var=r.var, cond=c, person=pron,
                             register=reg, codes=r.codes, reuse_existing=(c == "A"), system=system, user=user))
O = pd.DataFrame(out_rows)
O.to_csv(OUT / "prompts_register_v1_draft.csv", index=False, encoding="utf-8")

# ── (4) 검수용 대조표 ───────────────────────────────
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402

wb = Workbook()
ws = wb.active
ws.title = "도입부 대조표"
head = ["lead_id", "적용 문항 수", "적용 문항", "A 당신·하십시오체 (원문)", "B 당신·해라체", "B 출처", "B 자동점검",
        "C 이 사람·하십시오체", "C 출처", "C 자동점검", "D 이 사람·해라체", "D 출처", "D 자동점검",
        "검수자", "검수 결과(OK/수정)", "수정 제안", "메모"]
ws.append(head)
for _, r in leads.iterrows():
    ws.append([r.lead_id, int(r.n_items), r.vars, r.A, r.B, r.B_src, r.B_check, r.C, r.C_src, r.C_check,
               r.D, r.D_src, r.D_check, "", "", "", r.B_note or r.C_note or r.D_note])
widths = [11, 7, 22, 48, 48, 6, 18, 48, 6, 18, 48, 6, 18, 8, 10, 30, 30]
for i, w in enumerate(widths):
    ws.column_dimensions[chr(65 + i)].width = w
hdr = PatternFill("solid", fgColor="DDDDDD")
warn = PatternFill("solid", fgColor="FCE4D6")
corr_fill = PatternFill("solid", fgColor="E2EFDA")
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = hdr
for row in ws.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col_src, col_chk in ((5, 6), (8, 9), (11, 12)):
        if row[col_src].value == "보정":
            row[col_src - 1].fill = corr_fill
        if row[col_chk].value:
            row[col_chk].fill = warn
ws.freeze_panes = "E2"

ws2 = wb.create_sheet("고정 문구")
ws2.append(["부분", "A 당신·하십시오체", "B 당신·해라체", "C 이 사람·하십시오체", "D 이 사람·해라체"])
ws2.append(["system"] + [SYSTEM[CONDS[c][1]] for c in "ABCD"])
ws2.append(["페르소나 (L0만)"] + [PERSONA[c] for c in "ABCD"])
ws2.append(["마지막 지시문"] + [INSTR[c] for c in "ABCD"])
ws2.append(["c0에서 쓰는 조건", "O", "O", "-", "-"])
for i, w in enumerate([16, 42, 42, 42, 42]):
    ws2.column_dimensions[chr(65 + i)].width = w

ws3 = wb.create_sheet("전체 프롬프트 예시")
ws3.append(["probe_id", "cond", "system", "user"])
ex_vars = ["SUCDPAED", "CORRUPTN", "CESD1"]
for v in ex_vars:
    for _, r in O[(O["var"] == v)].iterrows():
        ws3.append([r.probe_id, r.cond, r.system, r.user])
ws3.column_dimensions["A"].width = 24
ws3.column_dimensions["C"].width = 40
ws3.column_dimensions["D"].width = 90
for row in ws3.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
wb.save(OUT / "review_table.xlsx")

leads.to_csv(OUT / "lead_table.csv", index=False, encoding="utf-8-sig")

summary = {
    "leads": len(leads), "items": P["var"].nunique(),
    "prompts_total": len(O), "prompts_new": int((~O.reuse_existing).sum()),
    "by_level_cond": O.groupby(["level", "cond"]).size().to_dict(),
    "corrected": {c: int((leads[f"{c}_src"] == "보정").sum()) for c in "BCD"},
    "flagged": {c: int((leads[f"{c}_check"] != "").sum()) for c in "BCD"},
    "options_with_honorific": len(opt_hon), "statements_with_honorific": len(stmt_hon),
    "A_matches_original": True,
}
print(summary)
for c in "BCD":
    bad = leads[leads[f"{c}_check"] != ""]
    for _, r in bad.iterrows():
        print(f"[{c}] {r.lead_id}: {r[f'{c}_check']} | {r[c]}")
pd.Series({str(k): v for k, v in summary.items()}).to_json(OUT / "summary.json", force_ascii=False, indent=1)
