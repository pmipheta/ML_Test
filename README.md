# ML_Test — ชุดโปรเจกต์ Machine Learning สำหรับเรียนรู้/ส่งงาน

3 โฟลเดอร์ย่อย แต่ละตัวทำงานเป็นอิสระจากกัน:

| โฟลเดอร์ | ทำอะไร | เริ่มจาก |
|---|---|---|
| **`regression/`** | ทำนายค่าต่อเนื่อง (ราคา, ดอกเบี้ย) — โมเดลเขียนเองด้วย numpy 3 ตัว | `python my_run.py` (คู่มือใน README.md ของโฟลเดอร์) |
| **`classification/`** | ทำนายหมวด (yes/no ฯลฯ) — 10 โมเดล sklearn | `python my_run.py` |
| **`classification_use/`** | Notebook pipeline 2 ขั้น: classification ⇄ regression (สลับทิศทางได้) | เปิด `classification_then_regression.ipynb` |

**ต้องมี:** Python 3.13+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `xgboost` (แนะนำ `lightgbm`)

**จุดเด่นของทั้ง 3 โปรเจกต์:**
- ทำความสะอาดข้อมูลสกปรกอัตโนมัติ (duplicates / inf / NaN / ข้อความตัวเลข / ID / outlier)
- กัน data leakage ด้วยหลักการ: split ก่อน, fit บน train เท่านั้น, out-of-fold prediction
- มีตัวตรวจ leak/overfit (`leak_check.py`, self-check ใน notebook)
- มีคู่มือการใช้งานละเอียดในแต่ละโฟลเดอร์ (`README.md`)
