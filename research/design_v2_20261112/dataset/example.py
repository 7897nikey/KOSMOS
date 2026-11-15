"""silicon.py 사용 예. 실행: python3 example.py  (pandas·numpy만 필요, 수 초)"""
import silicon as sl

df = sl.load()                                   # test 49문항 제외, long DataFrame
beta = sl.fit_beta(df)                           # calibration 148문항에서 모델×단계별 β*
print("β*\n", {k: float(v) for k, v in beta.items()})

val = df[df["split"] == "validation"]
print("\nvalidation TVD (β*)\n", sl.summary(val, beta).round(3))
print("\nvalidation TVD (β=1, 평활만)\n", sl.summary(val, 1.0).round(3))
print("\n균등분포 기준선\n", sl.uniform_baseline(val).iloc[:1].round(3))

# 직접 만지기: 문항 하나의 모델 분포와 사람 분포
x = sl.calibrate(df[(df["var"] == "ACPTCULT") & (df["persona"] == "L0")], 0.35)
print("\n", x[["model", "code", "count", "p", "human_p"]].round(3).to_string(index=False))
