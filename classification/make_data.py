"""สร้าง data.csv สำหรับงาน classification (ข้อมูลยื่นกู้ ที่มีความไม่สะอาดแบบข้อมูลจริง)"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
n = 400

customer_id = []
index = 1
while index <= n:
    customer_id.append("CUST" + str(index).zfill(4))
    index += 1

age = np.round(rng.normal(38, 9, n), 1)
credit_score = np.clip(rng.normal(660, 70, n), 300, 850)
income_kthb = np.round(np.abs(rng.normal(45, 20, n)) + 8, 1)
employment_years = np.round(np.abs(rng.normal(6, 4, n)), 1)
loan_amount = np.round(np.abs(rng.normal(320, 150, n)) + 20, 1)
purpose = rng.choice(["car", "home", "study", "business"], n)
channel = rng.choice(["branch", "online", "agent"], n)
applied_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")

# ความน่าจะเป็นอนุมัติ มาจาก score / income / loan / employment (แบบ logistic)
score_part = (credit_score - 630) / 90.0
income_part = (income_kthb - 40) / 25.0
loan_part = (loan_amount - 300) / 200.0
job_part = (employment_years - 5) / 8.0
logit = -0.2 + 1.35 * score_part + 0.65 * income_part - 0.55 * loan_part + 0.3 * job_part
probability = 1.0 / (1.0 + np.exp(-logit))
approved = np.where(rng.uniform(0, 1, n) < probability, "yes", "no")

df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "credit_score": credit_score,
    "income_kthb": income_kthb,
    "employment_years": employment_years,
    "loan_amount": loan_amount,
    "purpose": purpose,
    "channel": channel,
    "applied_date": applied_date,
    "constant_col": "x",
    "approved": approved,
})

# 1) NaN 5 แถว
df.loc[[3, 41, 99, 210, 377], "credit_score"] = np.nan
# 2) ค่า inf 2 แถว
df.loc[[7, 288], "income_kthb"] = np.inf
# 3) ตัวเลขที่เก็บเป็นข้อความ + มี comma
text_index = [12, 55, 130, 244, 301]
for position in text_index:
    raw = df.loc[position, "income_kthb"]
    df.loc[position, "income_kthb"] = f"{raw:,.1f}"
# 4) ตัวเลขที่เก็บเป็นข้อความล้วน
for position in [18, 175]:
    df.loc[position, "age"] = str(df.loc[position, "age"])
# 5) แถวซ้ำ 8 แถว
duplicates = df.sample(8, random_state=7)
df = pd.concat([df, duplicates], ignore_index=True)

df.to_csv("data.csv", index=False, encoding="utf-8-sig")
print("เขียน data.csv:", df.shape)
print("approved:", df["approved"].value_counts().to_dict())
print("คอลัมน์:", list(df.columns))
