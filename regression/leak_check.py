"""
leak_check.py — ตรวจ data leakage + overfit แบบเต็มระบบ (โปรเจกต์ regression)

วิธีใช้:
    python leak_check.py                          # ใช้ CSV_PATH/TARGET ด้านล่าง
    python leak_check.py loan_data.csv int.rate   # ระบุไฟล์/target ตรง ๆ

ตรวจ 6 ชั้น:
 1) แถวซ้ำทั้งแถว
 2) แถวรั่วข้าม train/test (contamination — แถวเดียวกันตกสองฝั่ง)
 3) feature ที่เป็น "สำเนาของ target" (|corr| ใกล้ 1)
 4) ablation ตัดทีละ feature — ตัวไหนแบกคะแนนทั้งโมเดล (วินิจฉัย leak)
 5) prune candidate — ตัวไหนควรตัด โดยตัดสินจาก k-fold CV บน train เท่านั้น
    (ใช้ test ตัดสินใจเรื่องเลือก feature = อีกรูปแบบของ leak จึงห้าม)
 6) overfit — เทียบ R2 train กับ R2 test
สุดท้ายพิมพ์ verdict PASS/WARN ทุกข้อ + คำแนะนำที่ทำตามได้ทันที
"""

import sys
from collections import Counter

import numpy as np

from my_run import load_csv, prepare_data
from multiple_linear_regression import MultipleLinearRegression

# ============ CONFIG (แก้ตรงนี้ หรือส่งผ่าน command line) ============
CSV_PATH = "data.csv"
TARGET = "price_thousand_thb"
TEST_SIZE = 0.2
SEED = 42          # ต้องเป็น seed เดียวกับที่ my_run ใช้ (จะได้ split ตรงกัน)
KFOLDS = 5

LEAK_CORR = 0.95   # |corr| กับ target เกินนี้ = สงสัยเป็นสำเนา/ฟังก์ชันของ target
BIG_DROP = 0.05    # ablation: ตัดแล้ว test R2 ตกเกินนี้ = feature แบกคะแนนมาก
PRUNE_GAIN = 0.005 # CV บน train: ตัดแล้ว CV R2 ดีขึ้นเกินนี้ = ควรพิจารณาตัด
OVERFIT_GAP = 0.10 # train R2 - test R2 เกินนี้ = สงสัย overfit
# ================================

# รับไฟล์/target จาก command line (ถ้ามี)
if len(sys.argv) >= 3:
    CSV_PATH, TARGET = sys.argv[1], sys.argv[2]
    print("ใช้ไฟล์:", CSV_PATH, "| target:", TARGET)

df = load_csv(CSV_PATH)
X, y, names, info = prepare_data(df, TARGET)
n = len(X)
print("rows:", info["rows_used"], "/", info["rows_original"], "| features:", len(names))
print()


# ---------------------------------------------------------------------------
# helper: fit multiple regression แล้วคืน (R2_train, R2_test)
# ---------------------------------------------------------------------------
def eval_split(Xa, ya, name_list):
    m = MultipleLinearRegression(test_size=TEST_SIZE, seed=SEED)
    m.set_data(Xa, ya, name_list)
    m.split()
    w = m.fit_normal_equation()
    tr = m.metrics(m.y_train, m.predict(m.X_train, w))["r2"]
    te = m.metrics(m.y_test, m.predict(m.X_test, w))["r2"]
    return tr, te


def cv_r2_train_only(Xa, ya, k=KFOLDS):
    """k-fold CV บนข้อมูลชุดนี้ทั้งชุด — ใช้ตัดสินเรื่องเลือก feature แบบไม่แตะ test"""
    nn = len(Xa)
    rs = np.random.RandomState(SEED)
    folds = np.array_split(rs.permutation(nn), k)
    scores = []
    for f in folds:
        mask = np.ones(nn, dtype=bool)
        mask[f] = False
        m = MultipleLinearRegression()
        m.set_data(Xa[mask], ya[mask])
        w = m.fit_normal_equation(m.X, m.y)
        pred = m.predict(Xa[f], w)
        scores.append(m.metrics(ya[f], pred)["r2"])
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 1-2) แถวซ้ำ + แถวรั่วข้าม train/test
# ---------------------------------------------------------------------------
rows_full = np.column_stack([X, y])
row_key = [rows_full[i].tobytes() for i in range(n)]

n_dup_rows = n - len(set(row_key))

rs = np.random.RandomState(SEED)
shuffled = rs.permutation(n)
n_train = int((1.0 - TEST_SIZE) * n)
train_idx, test_idx = shuffled[:n_train], shuffled[n_train:]

train_keys = set(row_key[i] for i in train_idx)
overlap = sum(1 for i in test_idx if row_key[i] in train_keys)

print("=" * 64)
print("1) แถวซ้ำ / แถวรั่วข้าม train/test")
print("=" * 64)
print("   แถวซ้ำทั้งแถว (หลัง prepare):", n_dup_rows)
print("   แถวเดียวกันตกทั้ง train และ test:", overlap)
print()


# ---------------------------------------------------------------------------
# 3) feature ที่เป็นสำเนาของ target
# ---------------------------------------------------------------------------
print("=" * 64)
print("2) corr ของแต่ละ feature กับ target")
print("=" * 64)
corr = []
for i in range(X.shape[1]):
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(X[:, i], y)[0, 1])
    corr.append(0.0 if not np.isfinite(r) else r)

leak_candidates = []
for i, nm in enumerate(names):
    mark = ""
    if abs(corr[i]) > LEAK_CORR:
        mark = "   <-- สงสัยเป็นสำเนา target!"
        leak_candidates.append(nm)
    print(f"   {nm:24s} corr = {corr[i]:8.4f}{mark}")
print()


# ---------------------------------------------------------------------------
# 4) ablation ตัดทีละ feature (วินิจฉัย: ตัวไหนแบกคะแนน)
# ---------------------------------------------------------------------------
tr_r2, te_r2 = eval_split(X, y, names)
print("=" * 64)
print("3) ตัวแบบทั้งหมด:  train R2 = %.4f   test R2 = %.4f" % (tr_r2, te_r2))
print("=" * 64)
print()
print("ตัดทีละ feature -> test R2 เหลือเท่าไร (drop มาก = แบกคะแนนมาก):")
big_drop = []
for i, nm in enumerate(names):
    keep = [j for j in range(X.shape[1]) if j != i]
    X2 = X[:, keep]
    names2 = [names[j] for j in keep]
    _, te2 = eval_split(X2, y, names2)
    drop = te_r2 - te2
    flag = ""
    if drop > BIG_DROP:
        flag = "   <-- ตกฮวบ! ตรวจว่าเป็นฟังก์ชันของ target หรือเปล่า"
        big_drop.append((nm, drop))
    print(f"   ตัด {nm:22s} -> test R2 = {te2:.4f}  (drop {drop:+.4f}){flag}")
print()


# ---------------------------------------------------------------------------
# 5) prune candidate — ตัดสินด้วย CV บน train เท่านั้น
# ---------------------------------------------------------------------------
print("=" * 64)
print("4) ควรตัด feature ไหน? (ตัดสินด้วย k-fold CV บนข้อมูลฝั่งเรียนเท่านั้น)")
print("=" * 64)
base_cv = cv_r2_train_only(X, y)
print("   CV R2 แบบเก็บทุก feature  = %.4f" % base_cv)

prune_list = []
for i, nm in enumerate(names):
    keep = [j for j in range(X.shape[1]) if j != i]
    cv2 = cv_r2_train_only(X[:, keep], y)
    gain = cv2 - base_cv
    if gain > PRUNE_GAIN:
        prune_list.append((nm, gain, cv2))

if prune_list:
    print("   ตัดแล้ว CV ดีขึ้น (แนะนำให้ตัด):")
    for nm, gain, cv2 in prune_list:
        print(f"     - {nm:22s} CV {base_cv:.4f} -> {cv2:.4f} (+{gain:.4f})")
else:
    print("   ไม่มี feature ที่ตัดแล้ว CV ดีขึ้น -> เก็บไว้ทั้งหมด")
print()


# ---------------------------------------------------------------------------
# 6) verdict รวม
# ---------------------------------------------------------------------------
print("=" * 64)
print("5) VERDICT รวม")
print("=" * 64)
checks = [
    ("ไม่มีแถวซ้ำหลังเตรียมข้อมูล", n_dup_rows == 0,
     f"duplicates = {n_dup_rows}"),
    ("train/test ไม่มีแถวรั่วข้ามกัน", overlap == 0,
     f"overlap = {overlap} แถว"),
    ("ไม่มี feature ที่ |corr| > %.2f กับ target" % LEAK_CORR,
     len(leak_candidates) == 0, f"พบ: {leak_candidates}" if leak_candidates else "ไม่พบ"),
    ("ไม่มีสัญญาณ overfit (train - test < %.2f)" % OVERFIT_GAP,
     (tr_r2 - te_r2) < OVERFIT_GAP,
     f"train = {tr_r2:.4f}, test = {te_r2:.4f}, gap = {tr_r2 - te_r2:.4f}"),
]
for label, ok, detail in checks:
    print(f"[{'PASS' if ok else 'WARN'}] {label}")
    print(f"        -> {detail}")

print()
print("คำแนะนำ:")
if leak_candidates:
    print("  1. ตัด feature ที่ |corr| ~ 1 ออกทันที (เป็นสำเนา target):", leak_candidates)
if big_drop:
    print("  2. feature ที่ตัดแล้วคะแนนตกฮวบ ต้องตรวจด้วยความรู้ธุรกิจว่าเป็น"
          " 'สาเหตุของ target' (เก็บไว้) หรือ 'ฟังก์ชัน/ผลลัพธ์ของ target' (leak ตัดทิ้ง):")
    for nm, drop in big_drop:
        print(f"       - {nm} (drop {drop:+.4f})")
if prune_list:
    print("  3. ตัด feature ที่ CV บน train บอกว่าไม่ช่วย (ทำแล้วรัน my_run ซ้ำได้เลย):")
    for nm, gain, cv2 in prune_list:
        print(f"       - {nm}")
if not (leak_candidates or big_drop or prune_list):
    print("  - ไม่พบปัญหา ข้อมูลและโมเดลใช้ได้ตามที่เป็น")

print()
print("หมายเหตุ: ข้อ 2 ไม่สามารถตัดสินอัตโนมัติได้ 100% — ยกตัวอย่างเคส loan_data.csv")
print("installment คำนวณจาก int.rate (target) จึงเป็น leak ที่ต้องตัด")
print("แต่ fico เป็น 'สาเหตุจริง' ของดอกเบี้ย จึงเก็บไว้ — ตัดสินด้วยความรู้ธุรกิจเสมอ")
