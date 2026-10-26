# 📓 คู่มือการใช้งานโฟลเดอร์ classification_use (Notebook 2-Stage)

> **โฟลเดอร์นี้คืออะไร**: Notebook `classification_then_regression.ipynb` —
> pipeline สองขั้นที่ **สลับทิศทางได้**: ทำนาย class → ทำนายค่าตัวเลข หรือ ทำนายค่าตัวเลข → ทำนาย class
> ใช้ sklearn + XGBoost + LightGBM + Stacking (14 ตระกูล) + ตรวจ leak/overfit อัตโนมัติ

---

## 📂 ไฟล์ในโฟลเดอร์

| ไฟล์ | หน้าที่ |
|---|---|
| `classification_then_regression.ipynb` | **Notebook หลัก** — pipeline สอง stage ทั้ง 2 ทิศทาง (35 เซลล์ รันแล้วมี output) |
| `classification_then_regression_ORIGINAL_backup.ipynb` | ต้นฉบับก่อนแก้ (สำรองไว้เทียบ) |
| `data.csv` | ข้อมูลของคุณ (loan default 800 แถว: class = `default`, reg = `loan_amount`) |
| `cls_reg_submission_skeleton.ipynb` / `regression_*` | เทมเพลตส่งงาน (ไม่เกี่ยวกับ pipeline หลัก) |

---

## 🚀 เริ่มใช้งาน 5 ขั้น

### ขั้นที่ 1 — เปิด notebook

- VS Code → Open Folder โฟลเดอร์นี้ → เปิด `classification_then_regression.ipynb`
- หรือ Jupyter: `jupyter notebook` แล้วเปิดไฟล์
- ต้องมี: `python`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost` (ถ้าอยากได้ LightGBM ด้วย: `pip install lightgbm`)

### ขั้นที่ 2 — วางข้อมูล

ไฟล์ CSV วางไว้ในโฟลเดอร์นี้ แล้วแก้เซลล์ที่ 2 (แผนก `## 0. โหลดข้อมูล`):

```python
DATA_PATH = 'data.csv'   # ← แก้เป็นชื่อไฟล์ของคุณ
df = pd.read_csv(DATA_PATH)
```

> หรือถ้ามี DataFrame อยู่แล้วในหน่วยความจำ: `df = your_dataframe.copy()` ได้เลย

### ขั้นที่ 3 — แก้ CONFIG (เซลล์ที่ 4 — บล็อก `# ============ CONFIG (แก้ตรงนี้) ============`)

```python
CLASS_COL = 'default'         # ← คอลัมน์ class (yes/no หรือหมวด)
REG_COL   = 'loan_amount'     # ← คอลัมน์ค่าต่อเนื่อง (ตัวเลข)
DROP_COLS = ['app_id']        # ← คอลัมน์ ID/ชื่อ/วันที่ ที่ไม่ควรใช้ (สำคัญ!)

DIRECTION  = 'cls2reg'        # ← สลับทิศทางที่นี่: 'cls2reg' หรือ 'reg2cls'
```

**ตารางอธิบายทุกตัวแปรใน CONFIG:**

| ตัวแปร | ค่า default | ความหมาย / ควรตั้งเมื่อไหร่ |
|---|---|---|
| `CLASS_COL` | `'default'` | คอลัมน์คำตอบแบบหมวด (2 คลาสขึ้นไป) |
| `REG_COL` | `'loan_amount'` | คอลัมน์คำตอบแบบตัวเลข |
| `DROP_COLS` | `['app_id']` | คอลัมน์ ID ที่ **ต้องตัด** ไม่งั้นโมเดลจะใช้เลขแถวเป็นความรู้ (leak ปลอม) |
| `DIRECTION` | `'cls2reg'` | **`'cls2reg'`** = Stage1 จำ class → Stage2 ทำนายค่า • **`'reg2cls'`** = กลับกัน |
| `TEST_SIZE` / `SEED` / `N_SPLITS` | 0.2 / 42 / 5 | สัดส่วน test, seed ทั้งระบบ, จำนวน fold |
| `MAX_NULL_COL` | 0.50 | คอลัมน์ null เกิน 50% → ตัดคอลัมน์ |
| `ADD_INDICATOR` | 0.05 | คอลัมน์ null เกิน 5% → เพิ่ม flag `_missing` |
| `MAX_CARD` | 20 | คอลัมน์ข้อความเกิน 20 ค่า (ID-like) → ตัด |
| `CLS_METRIC` | `'f1_macro'` | metric ที่ใช้เลือกโมเดล: `'accuracy'` / `'f1_macro'` / `'f1_weighted'` |
| `CLASS_WEIGHT` | `None` | คลาสไม่สมดุลมาก → ตั้ง `'balanced'` |
| `USE_LOG_TARGET` | `False` | y เบ้มาก (skew > 1) → `True` (regress log1p(y)) |
| `ADD_PROBA` / `ADD_CLASS_ONEHOT` | True/True | เอา probability / class ทำนาย มาเป็น feature ของ Stage 2 (cls2reg) |
| `ADD_PRED_VALUE` / `ADD_PRED_BIN` | True/True | เอาค่าทำนาย / bin มาเป็น feature ของ Stage 2 (reg2cls) |
| `WINSORIZE` (+LO/HI) | True (1%/99%) | ตัดหาง outlier ของ numeric (fit จาก train เท่านั้น) — ถ้าอยากได้คะแนนตามข้อมูลดิบตั้ง `False` |

### ขั้นที่ 4 — รันทั้งหมดตามลำดับ

**Run All** หรือรันเซลล์ทีละก้อนตามลำดับ:

| เซลล์ | แผนก | ทำอะไร |
|---|---|---|
| 2 | 0. โหลดข้อมูล | อ่าน CSV |
| 4–5 | 1. Config + ตรวจ | **ตั้งค่าทั้งหมดที่นี่** + ดู null/คลาส/skew ของข้อมูล |
| 7 | 2. Data prep | ทำความสะอาด + split แบบ stratified |
| 10–14 | 3. Stage 1 | เปรียบเทียบโมเดล (CV) → tune → วัดบน test |
| 16 | 4. สร้าง feature | **out-of-fold prediction** (กัน leak) ให้ Stage 2 |
| 19–23 | 5. Stage 2 | เปรียบเทียบ → ablation → tune → วัดบน test |
| 25–26 | 6. end-to-end | `predict_all()` ทดสอบกับ test + กราฟ |
| 29–31 | 7. ทางเลือก | per-group models + permutation importance |
| 33 | **8. Self-check** | **ตรวจ leak + overfit อัตโนมัติ — ต้อง PASS หมดก่อนใช้ผล** |

### ขั้นที่ 5 — สลับทิศทาง (จุดเด่นของ notebook นี้)

แก้บรรทัดเดียวแล้ว Run All ใหม่:

```python
DIRECTION = 'cls2reg'   # class → reg   (จำ class แล้วช่วยทำนายค่า)
DIRECTION = 'reg2cls'   # reg → class   (จำค่าแล้วช่วยจำ class)
```

---

## 🧹 ข้อมูลสกปรกที่รองรับอัตโนมัติ (ไม่ต้องแก้ไฟล์ก่อน)

| ปัญหา | จัดการว่า |
|---|---|
| แถวซ้ำ / target missing | ตัดแถว |
| `NaN` ใน numeric | median impute (fit จาก train เท่านั้น) |
| `NaN` ใน categorical | แทนเป็น `'missing'` |
| `inf` | แปลงเป็น missing |
| ข้อความ marker `'na' 'n/a' 'null' '-' '?'` | ถือเป็น missing |
| ตัวเลขเก็บเป็นข้อความ `"51,598"` `"45%"` `"$1,200"` | แปลงเป็นตัวเลข |
| คอลัมน์ datetime / ค่าคงที่ / ID เกิน 20 ค่า / ชื่อซ้ำ | ตัดคอลัมน์ |
| outlier | winsorize 1%/99% (skip คอลัมน์ 0/1) |
| คลาสน้อยเกินตอน CV | สลับเป็น KFold ให้เอง |
| category ใหม่ตอน predict ข้อมูลใหม่ | `handle_unknown='ignore'` ไม่พัง |

---

## ✅ วิธีอ่าน Self-check (เซลล์ section 8)

| ข้อ | วัดว่า | ถ้า WARN แปลว่า |
|---|---|---|
| แถวซ้ำ / รั่วข้าม train-test | ความสะอาดของ split | ข้อมูลต้องทำความสะอาดก่อน |
| สัดส่วน class train≈test | stratify ทำงาน | คลาสใดถึงขั้นขาด — เพิ่มข้อมูล |
| feature ไม่มี NaN/inf | feature ปลอดภัยสำหรับโมเดล | ดูมีคอลัมน์ไหนพัง |
| Stage1 fit − OOF < 0.15 | โมเดลไม่ "ท่องจำ" train | overfit Stage 1 → ลดความซับซ้อน |
| Stage1 OOF ≈ test | ทำนายบน train ใช้กับ test ได้จริง | ข้อมูลไม่เหมือนกัน / split โชคร้าย |
| Stage2 train − test < 0.15 | ไม่ overfit รอบสอง | ลดความซับซ้อน Stage 2 |

> ผลล่าสุด: `cls2reg` PASS 8/8 • `reg2cls` PASS 7/8 (1 WARN: OOF 0.661 vs test 0.550 — เกิดจาก winsorize + test เล็ก 160 แถว ถ้าอยากเลี่ยงตั้ง `WINSORIZE = False`)

---

## 🔮 ใช้กับข้อมูลใหม่จริง (เซลล์ 6)

```python
res = predict_all(new_df)   # new_df = dataframe ที่มีคอลัมน์ feature ครบ (ไม่ต้องมี target)
# คืน: pred_class (คลาสทำนาย) + proba_ทุกคลาส + pred_value (ค่าทำนาย)
res[['pred_class', 'pred_value']].to_csv('predictions.csv', index=False)   # บันทึกส่งงาน
```

---

## 📊 ผลอ้างอิงล่าสุด (data.csv: default/loan_amount)

**cls2reg**: Stage1 (classify) NaiveBayes F1-macro 0.717 | Stage2 (regress) Ridge R² 0.658 (CV), 0.548 (test) — end-to-end R² 0.548
**reg2cls**: Stage1 (regress) Ridge R² 0.658 (CV) | Stage2 (classify) Logistic F1 0.715 (CV) — end-to-end Acc 0.775 / F1 0.656
**ข้อสังเกต**: ablation แสดงว่า Stage 1 แทบไม่ช่วย Stage 2 (eta² = 0.043 — สอง target แทบไม่เกี่ยวกันในข้อมูลชุดนี้) — เป็นคุณสมบัติของข้อมูล ไม่ใช่บั๊ก

---

## ❓ ปัญหาที่เจอบ่อย

| อาการ | วิธีแก้ |
|---|---|
| `ไม่พบคอลัมน์ ...` ตอน assert | แก้ `CLASS_COL`/`REG_COL` ให้ตรงชื่อคอลัมน์จริง |
| Cell 16 error: `get_feature_names_out` | ใช้เวอร์ชันเก่าของไฟล์ — ตัวปัจจุบันแก้แล้ว (Winsorizer มี get_feature_names_out) |
| LightGBM ไม่ขึ้นในตาราง | `pip install lightgbm` แล้ว Run All ใหม่ (ไม่มีก็ข้ามให้อัตโนมัติ ไม่พัง) |
| ablation: "ทำนาย ≈ ไม่มี" | Stage 1 ยังไม่คม — tune Stage 1 ก่อน (เพิ่ม grid / เปลี่ยน CLS_METRIC) |
| เอา notebook ไปใช้กับไฟล์ใหม่ | แก้ 3 จุด: `DATA_PATH` (เซลล์ 2) + `CLASS_COL/REG_COL/DROP_COLS` + `DIRECTION` (เซลล์ 4) แล้ว Run All |

---

*อัปเดตล่าสุด: รันจริงทั้ง 2 ทิศทาง 0 error, self-check ผ่าน, รองรับข้อมูลสกปรกครบ (duplicates/inf/NaN/markers/ตัวเลขเป็นข้อความ/ID/outlier/unknown category)*
