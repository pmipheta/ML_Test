"""
ไฟล์เดียวที่ต้องแก้และรัน — ทำ 3 อย่าง:
    1) อ่าน DataFrame ของอาจารย์ (จาก CSV หรือส่ง df เข้ามา)
    2) ทำความสะอาดข้อมูลให้พร้อมใช้ (NaN / ข้อความ / วันที่ / ตัวเลขที่เก็บเป็นข้อความ)
    3) รัน 3 โมเดล แล้วพิมพ์ตาราง R2 / MSE / MAE + บอกโมเดลที่ดีสุดในแต่ละด้าน

วิธีใช้:
    1) วางไฟล์ CSV ไว้โฟลเดอร์เดียวกับไฟล์นี้ (หรือใส่ path เต็ม)
    2) แก้ 2 บรรทัดล่าง: CSV_PATH และ TARGET
    3) รัน:  python my_run.py

ถ้าต้องส่งอาจารย์แค่โมเดลเดียว:
    - เก็บไว้แค่ my_run.py + ไฟล์โมเดลตัวที่ต้องการ + data.csv
    - ตั้ง MODELS = ["multiple"] (หรือชื่อไฟล์ที่เหลือ)
    โปรแกรมจะรันเฉพาะโมเดลที่มีไฟล์อยู่ ไม่พังถ้าไฟล์อื่นหาย

ใน Jupyter:
    import pandas as pd
    from my_run import compare
    compare(pd.read_csv("data.csv"), target="price")
"""

import sys

import numpy as np
import pandas as pd

# --- โหลดโมเดลเท่าที่มีในโฟลเดอร์ ---
# ถ้าไฟล์ไหนไม่มี ระบบจะข้ามโมเดลนั้นไป ไม่ทำให้ทั้งโปรแกรมพัง
try:
    from linear_regression import LinearRegression
except ImportError:
    LinearRegression = None

try:
    from multiple_linear_regression import MultipleLinearRegression
except ImportError:
    MultipleLinearRegression = None

try:
    from polynomial_regression import PolynomialRegression
except ImportError:
    PolynomialRegression = None

# ===================== แก้ตรงนี้ =====================
CSV_PATH = "data.csv"             # ไฟล์ข้อมูลของอาจารย์
TARGET = "price_thousand_thb"     # ชื่อคอลัมน์ที่ต้องการทำนาย

# ถ้าส่งอาจารย์แค่บางโมเดล ให้ใส่เฉพาะชื่อที่มี เช่น MODELS = ["multiple"]

MODELS = ["linear", "multiple", "polynomial"]
# ====================================================

TEST_SIZE = 0.2             # เก็บ 20% ไว้ทดสอบ
SEED = 42                   # seed เดียวกันทุกโมเดล (เทียบกันยุติธรรม)
MAX_TEXT_CATEGORIES = 15    # คอลัมน์ข้อความที่มีหมวดเกินนี้จะถูกข้าม
OUTLIER_FACTOR = 1.5         # ตัวคูณ IQR (1.5 = มาตรฐาน, 3.0 = ตัดเฉพาะค่าสุดโต่งมาก)
OUTLIER_MAX_FRACTION = 0.15  # ถ้าจะตัดเกิน 15% ของข้อมูล = ไม่ตัด (ข้อมูลกว้างจริง)


# ---------------------------------------------------------------------------
# 1) อ่านไฟล์ CSV (ลอง encoding ให้เอง)
# ---------------------------------------------------------------------------
def load_csv(path):
    """
    อ่านไฟล์ CSV แล้วคืน DataFrame
    ลอง encoding ตามลำดับ: utf-8-sig (Excel), utf-8, cp874 (ภาษาไทยเก่า)

    Input:
        path: str ที่อยู่ไฟล์
    Output:
        pandas.DataFrame
    """
    encodings = ["utf-8-sig", "utf-8", "cp874"]

    last_error = None
    for encoding in encodings:
        try:
            dataframe = pd.read_csv(path, encoding=encoding)
            return dataframe
        except UnicodeDecodeError as error:
            last_error = error
        except OSError as error:
            # ไฟล์ไม่มี / path ผิด / เปิดไม่ได้ -> ข้อความไทย แทน traceback ยาว
            raise ValueError(
                "เปิดไฟล์ CSV ไม่ได้: " + str(error)
                + "\nตรวจว่าไฟล์อยู่ที่ " + str(path) + " ถูกต้อง"
            )

    message = "อ่านไฟล์ CSV ไม่ได้เพราะ encoding ไม่ตรง: " + str(last_error)
    message = message + "\nวิธีแก้: เปิดใน Excel แล้ว Save As -> 'CSV UTF-8'"
    raise ValueError(message)


# ---------------------------------------------------------------------------
# 2) แปลงคอลัมน์ข้อความให้เป็นตัวเลข (one-hot)
# ---------------------------------------------------------------------------
def encode_text_column(series, column_name, max_categories=MAX_TEXT_CATEGORIES):
    """
    แปลงคอลัมน์ข้อความเป็นคอลัมน์ตัวเลข 0/1

    กติกา:
        - ค่าคงที่ทั้งคอลัมน์      -> ข้าม
        - 2 หมวด                  -> 1 คอลัมน์ (0/1)
        - 3..max_categories หมวด  -> one-hot โดยตัดหมวดแรกทิ้ง (k-1 คอลัมน์)
        - เกิน max_categories     -> ข้าม (น่าจะเป็น ID หรือชื่อ)
        - ค่าว่าง                  -> นับเป็นหมวด "__missing__"

    ทำไมต้องตัดหมวดแรก: ถ้าเก็บครบทุกหมวด ผลรวมของคอลัมน์จะเท่ากับ 1
    ซึ่งซ้ำกับ intercept ทำให้เมทริกซ์ XᵀX singular และคำนวณพัง

    Input:
        series: pandas.Series คอลัมน์ข้อความ
        column_name: str ชื่อคอลัมน์
        max_categories: int จำนวนหมวดสูงสุดที่ยอมให้ one-hot
    Output:
        (list ของ (ชื่อคอลัมน์ใหม่, Series 0/1), เหตุผลที่ข้าม หรือ None)
    """
    text = series.astype("string").fillna("__missing__")
    categories = list(pd.unique(text))

    if len(categories) <= 1:
        return [], "ค่าคงที่ทั้งคอลัมน์"

    if len(categories) > max_categories:
        reason = str(len(categories)) + " หมวด (เกิน " + str(max_categories) + ")"
        return [], reason

    parts = []
    index = 1
    while index < len(categories):
        category = categories[index]
        new_name = str(column_name) + "=" + str(category)
        values = (text == category).astype(float)
        parts.append((new_name, values))
        index += 1

    return parts, None


# ---------------------------------------------------------------------------
# 3.5) หา outlier ด้วยกฎ IQR
# ---------------------------------------------------------------------------
def inlier_mask(values, factor=OUTLIER_FACTOR):
    """
    คืน mask ว่าแถวไหน "อยู่ในช่วงปกติ" ตามกฎ IQR

    กฎ IQR (มาตรฐานของงานสถิติ):
        Q1  = ควอไทล์ที่ 25%
        Q3  = ควอไทล์ที่ 75%
        IQR = Q3 - Q1
        ค่าปกติ = อยู่ในช่วง [Q1 - factor*IQR , Q3 + factor*IQR]
        factor = 1.5 คือค่ามาตรฐาน (หลุดช่วงนี้ถือเป็น outlier)

    ตัวอย่าง: ข้อมูลส่วนใหญ่อยู่ 10-20 แต่มีค่า 900 โผล่มา
              -> 900 อยู่นอกช่วง จึงถูกทำเครื่องหมายว่าผิดปกติ

    Input:
        values: np.ndarray 1 มิติ (คอลัมน์เดียว)
        factor: float ตัวคูณ IQR (ยิ่งมาก ยิ่งตัดน้อย)
    Output:
        np.ndarray dtype=bool (True = อยู่ในช่วงปกติ)
    """
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1

    if np.isclose(iqr, 0.0):
        # ค่าส่วนใหญ่ซ้ำกันจน IQR = 0 -> ตัดสินไม่ได้ ไม่ตัดแถวไหนเลย
        return np.ones(len(values), dtype=bool)

    low = q1 - factor * iqr
    high = q3 + factor * iqr

    return (values >= low) & (values <= high)


# ---------------------------------------------------------------------------
# 4) ทำความสะอาดข้อมูล -> numpy arrays
# ---------------------------------------------------------------------------
def prepare_data(df, target, features=None):
    """
    แปลง DataFrame เป็น X, y พร้อมจัดการปัญหาที่เจอได้จริง

    สิ่งที่ทำให้อัตโนมัติ:
        - ตัดแถวที่มี NaN หรือ inf
        - ตัดคอลัมน์ค่าคงที่ และคอลัมน์ที่ว่างทั้งคอลัมน์
        - ข้อความ -> one-hot ; ตัวเลขที่เก็บเป็นข้อความ ("1500", "51,598") -> ตัวเลข
        - คอลัมน์วันที่ และข้อความที่มีหมวดเยอะเกินไป -> ข้าม

    Input:
        df: pandas.DataFrame
        target: str ชื่อคอลัมน์เป้าหมาย (ต้องเป็นตัวเลข)
        features: None = ใช้ทุกคอลัมน์ที่เหลือ, หรือ list[str] เลือกเอง
    Output:
        X (n, p), y (n,), feature_names list[str], info dict
    """
    if df is None:
        raise ValueError("ไม่ได้ส่ง DataFrame มา")

    work = df.copy()

    # --- ตัดแถวซ้ำทั้งแถว (กัน train/test contamination) ---
    # แถวที่ซ้ำกันเป๊ะถ้าตกคนละฝั่ง train/test จะทำให้คะแนนสูงเกินจริง
    n_dup = int(work.duplicated().sum())
    if n_dup > 0:
        work = work.drop_duplicates().reset_index(drop=True)
        print("[clean] ตัดแถวซ้ำออก:", n_dup, "แถว")

    # --- ตรวจชื่อคอลัมน์ซ้ำ ---
    duplicated = []
    for column in work.columns[work.columns.duplicated()]:
        duplicated.append(str(column))
    if len(duplicated) > 0:
        raise ValueError("DataFrame มีชื่อคอลัมน์ซ้ำ: " + str(duplicated))

    # --- ตรวจว่ามี target ---
    if target not in work.columns:
        available = []
        for column in work.columns:
            available.append(str(column))
            if len(available) >= 20:
                break
        raise ValueError(
            "ไม่พบคอลัมน์ target ชื่อ " + repr(target)
            + "\nคอลัมน์ที่มี: " + str(available)
        )

    # --- target ต้องเป็นตัวเลข ---
    y_series = pd.to_numeric(work[target], errors="coerce")
    if y_series.notna().sum() == 0:
        raise ValueError(
            "คอลัมน์ target " + repr(target) + " ไม่ใช่ตัวเลข"
            " — โปรเจกต์นี้ทำ regression จึงต้องมี target เป็นตัวเลข"
        )

    # --- เลือกคอลัมน์ที่จะใช้เป็น feature ---
    if features is None:
        feature_columns = []
        for column in work.columns:
            if column != target:
                feature_columns.append(column)
    else:
        selected = features
        if isinstance(features, str):
            selected = [features]

        feature_columns = []
        for column in selected:
            if column == target:
                continue
            already = False
            for existing in feature_columns:
                if existing == column:
                    already = True
                    break
            if not already:
                feature_columns.append(column)

        missing = []
        for column in feature_columns:
            if column not in work.columns:
                missing.append(column)
        if len(missing) > 0:
            raise ValueError("ไม่พบคอลัมน์ feature: " + str(missing))

    n_original = len(work)

    # --- วนทีละคอลัมน์: ตัวเลขใช้ตรง ๆ / ข้อความแปลงเป็น 0-1 ---
    part_series = []
    part_names = []

    for column in feature_columns:
        series = work[column]

        # 0) คอลัมน์ index ที่ pandas สร้างตอน save CSV (Unnamed: 0) -> ข้าม
        #    เพราะเป็นเลขแถว ไม่ใช่ข้อมูลจริง ใส่เข้าไปจะทำให้โมเดลจำแถว
        if str(column).startswith("Unnamed:"):
            continue

        # 1) คอลัมน์ตัวเลขหรือ boolean -> ใช้ค่าตรง ๆ
        if pd.api.types.is_numeric_dtype(series):
            part_series.append(pd.to_numeric(series, errors="coerce"))
            part_names.append(str(column))
            continue
        if pd.api.types.is_bool_dtype(series):
            part_series.append(pd.to_numeric(series, errors="coerce"))
            part_names.append(str(column))
            continue

        # 2) คอลัมน์วันที่ -> ข้าม
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        # 3) คอลัมน์ข้อความ: ลองแปลงเป็นตัวเลขก่อน ("1500" / "51,598")
        cleaned = series.astype("string")
        cleaned = cleaned.str.replace(",", "", regex=False)
        cleaned = cleaned.str.strip()

        non_null_count = int(series.notna().sum())
        as_number = pd.to_numeric(cleaned, errors="coerce")
        number_count = int(as_number.notna().sum())

        if non_null_count > 0 and number_count >= 0.9 * non_null_count:
            part_series.append(as_number)
            part_names.append(str(column))
            continue

        # 4) คอลัมน์วันที่ที่อ่านมาเป็นข้อความ -> ข้าม
        as_datetime = None
        try:
            as_datetime = pd.to_datetime(cleaned, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            as_datetime = None

        if as_datetime is not None and non_null_count > 0:
            datetime_count = int(as_datetime.notna().sum())
            if datetime_count >= 0.9 * non_null_count:
                continue

        # 5) ที่เหลือถือเป็นข้อความ -> one-hot
        encoded_parts = encode_text_column(series, str(column))[0]
        position = 0
        while position < len(encoded_parts):
            new_name = encoded_parts[position][0]
            values = encoded_parts[position][1]
            part_series.append(values)
            part_names.append(new_name)
            position += 1

    if len(part_series) == 0:
        raise ValueError("ไม่มีคอลัมน์ feature ที่ใช้ได้เลย")

    # --- รวมทุกคอลัมน์เป็น DataFrame เดียว ---
    renamed_parts = []
    position = 0
    while position < len(part_series):
        renamed_parts.append(part_series[position].rename(part_names[position]))
        position += 1
    X_frame = pd.concat(renamed_parts, axis=1)

    # --- ตัดคอลัมน์ที่เป็น NaN ทั้งคอลัมน์ ---
    empty_columns = []
    for column in X_frame.columns:
        if X_frame[column].isna().all():
            empty_columns.append(column)

    if len(empty_columns) > 0:
        X_frame = X_frame.drop(columns=empty_columns)
        kept_names = []
        for name in part_names:
            if name not in empty_columns:
                kept_names.append(name)
        part_names = kept_names

    if X_frame.shape[1] == 0:
        raise ValueError("ทุกคอลัมน์ feature เป็นค่าว่าง (NaN) ทั้งคอลัมน์")

    # --- ตัดแถวที่มี NaN หรือ inf ---
    # แปลงเป็น numpy แล้วเช็คทีเดียวทั้งเมทริกซ์
    # isfinite ครอบทั้ง NaN และ inf และเร็วกว่าการไล่เช็คทีละแถวด้วย iloc
    # หลายร้อยเท่า เมื่อข้อมูลมีหลักหมื่นแถวขึ้นไป
    X_values = X_frame.to_numpy(dtype=float)
    y_values = y_series.to_numpy(dtype=float)

    keep_mask = np.isfinite(X_values).all(axis=1)
    keep_mask = keep_mask & np.isfinite(y_values)

    X_values = X_values[keep_mask]
    y_values = y_values[keep_mask]

    # --- ตัดคอลัมน์ค่าคงที่ (std = 0) ---
    stds = np.std(X_values, axis=0)

    variable_columns = []
    column_index = 0
    while column_index < len(stds):
        if not np.isclose(stds[column_index], 0.0):
            variable_columns.append(column_index)
        column_index += 1

    if len(variable_columns) == 0:
        raise ValueError("หลังตัดคอลัมน์ค่าคงที่แล้วไม่เหลือ feature")

    X_values = X_values[:, variable_columns]
    final_names = []
    for column_index in variable_columns:
        final_names.append(part_names[column_index])

    # --- ตัด outlier ด้วยกฎ IQR ---
    # ทำกับทุก feature และ target: แถวไหนสุดโต่งที่คอลัมน์ใดคอลัมน์หนึ่ง = ตัดทั้งแถว
    # เหตุผล: ค่าเดียวที่หลุดไปไกลจะดึงเส้นถดถอยทั้งเส้นให้เอียง (least squares
    # ให้ความสำคัญกับ error ยกกำลังสอง แถวสุดโต่งจึงมีน้ำหนักมหาศาล)
    outlier_keep = np.ones(X_values.shape[0], dtype=bool)

    column_index = 0
    while column_index < X_values.shape[1]:
        outlier_keep = outlier_keep & inlier_mask(X_values[:, column_index])
        column_index += 1

    outlier_keep = outlier_keep & inlier_mask(y_values)

    flagged = int((~outlier_keep).sum())
    outliers_removed = 0
    outliers_skipped = 0

    if flagged > 0:
        fraction = flagged / X_values.shape[0]
        if fraction <= OUTLIER_MAX_FRACTION:
            # ตัดไม่มาก -> เชื่อว่าเป็นค่าผิดปกติจริง
            X_values = X_values[outlier_keep]
            y_values = y_values[outlier_keep]
            outliers_removed = flagged
        else:
            # จะตัดเกิน 15% แปลว่าข้อมูลมีการกระจายกว้างตามธรรมชาติ
            # (หรือใช้กฎเข้มเกินไป) การตัดจะทำให้เสียข้อมูลมาก -> เก็บไว้ทั้งหมด
            outliers_skipped = flagged

    # --- ตรวจจำนวนแถวและความแปรปรวนของ target ---
    if X_values.shape[0] < 10:
        raise ValueError(
            "เหลือข้อมูลใช้ได้ " + str(X_values.shape[0]) + " แถว (ต้องมีอย่างน้อย 10)"
        )

    if np.isclose(np.std(y_values), 0.0):
        raise ValueError(
            "คอลัมน์ target " + repr(target) + " มีค่าคงที่ทุกแถว ทำนายไม่ได้"
        )

    info = {}
    info["rows_original"] = n_original
    info["rows_used"] = int(X_values.shape[0])
    # นับแถวที่ซ้ำกันทั้งแถว (ข้อมูลจริงมักมี) เพื่อให้รู้ว่าข้อมูลมีปัญหาหรือไม่
    info["duplicate_rows"] = int(df.duplicated().sum())
    info["outliers_removed"] = outliers_removed
    info["outliers_skipped"] = outliers_skipped

    return X_values, y_values, final_names, info


# ---------------------------------------------------------------------------
# 4) รันทุกโมเดลแล้วเทียบ
# ---------------------------------------------------------------------------
def compare(df, target, features=None, test_size=TEST_SIZE, seed=SEED):
    """
    รัน 3 โมเดลบนข้อมูลชุดเดียวกัน แล้วพิมพ์ตาราง R2 / MSE / MAE

    Input:
        df: pandas.DataFrame
        target: str ชื่อคอลัมน์เป้าหมาย
        features: None หรือ list[str]
        test_size: float สัดส่วนข้อมูลทดสอบ
        seed: int seed สำหรับแบ่ง train/test (ใช้ร่วมกันทุกโมเดล)
    Output:
        list ของ dict ผลแต่ละโมเดล
    """
    X, y, feature_names, info = prepare_data(df, target, features)

    # --- เตรียมรายชื่อโมเดลที่จะรัน (ข้ามตัวที่ไม่มีไฟล์) ---
    known_names = ["linear", "multiple", "polynomial"]
    unknown_names = []
    for name in MODELS:
        if name not in known_names:
            unknown_names.append(name)
    if len(unknown_names) > 0:
        raise ValueError(
            "ชื่อโมเดลใน MODELS ไม่ถูกต้อง: " + str(unknown_names)
            + "\nชื่อที่ใช้ได้: " + str(known_names)
        )

    model_classes = []
    model_classes.append(("linear", LinearRegression))
    model_classes.append(("multiple", MultipleLinearRegression))
    model_classes.append(("polynomial", PolynomialRegression))

    models = []
    skipped = []
    for name, model_class in model_classes:
        # ไม่ได้เลือกใน MODELS -> ข้าม
        if name not in MODELS:
            continue
        # ไม่พบไฟล์โมเดล -> ข้าม (ไม่ทำให้ทั้งโปรแกรมพัง)
        if model_class is None:
            skipped.append(name + " (ไม่พบไฟล์)")
            continue
        models.append(model_class())

    if len(models) == 0:
        raise ValueError(
            "ไม่มีโมเดลให้รันเลย — ตรวจสอบว่ามีไฟล์โมเดลอยู่ในโฟลเดอร์ "
            "และชื่อใน MODELS ถูกต้อง"
        )

    # ตั้งค่าให้เหมือนกันทุกโมเดล (เทียบกันยุติธรรม) แล้วรัน
    results = []
    for model in models:
        model.test_size = test_size
        model.seed = seed

        # ถ้าโมเดลตัวใดตัวหนึ่งรันไม่ผ่าน ให้ข้ามตัวนั้นแล้วรันตัวอื่นต่อ
        try:
            summary = model.run(X, y, feature_names)
        except ValueError as error:
            skipped.append(model.__class__.__name__ + " (" + str(error) + ")")
            continue

        results.append(summary)

    if len(results) == 0:
        raise ValueError("รันทุกโมเดลไม่สำเร็จ: " + str(skipped))

    for message in skipped:
        print("[skip] " + message)

    # --- พิมพ์หัวตาราง ---
    header = "rows " + str(info["rows_used"]) + "/" + str(info["rows_original"])
    header = header + " | features " + str(len(feature_names))
    header = header + " | target " + str(target)
    if info.get("duplicate_rows", 0) > 0:
        header = header + " | duplicates " + str(info["duplicate_rows"])
    if info.get("outliers_removed", 0) > 0:
        header = header + " | outliers " + str(info["outliers_removed"])
    if info.get("outliers_skipped", 0) > 0:
        header = header + " | outliers skipped " + str(info["outliers_skipped"])
    print(header)
    print()
    print("Model                                R2_train    R2_test        MSE          MAE")
    print("-" * 82)

    for summary in results:
        model_name = summary["model"]
        line = model_name.ljust(34)
        r2_train = summary.get("r2_train")
        if r2_train is None:
            line = line + "      n/a"
        else:
            line = line + format(r2_train, ">9.4f")
        line = line + format(summary["r2"], ">9.4f")
        line = line + format(summary["mse"], ">13.4f")
        line = line + format(summary["mae"], ">13.4f")
        print(line)

    print("-" * 82)

    # --- เตือน overfit: train สูงกว่า test มาก = โมเดลจำ train ---
    for summary in results:
        r2_train = summary.get("r2_train")
        if r2_train is None:
            continue
        gap = r2_train - summary["r2"]
        if gap > 0.10:
            print("[overfit?] " + summary["model"]
                  + ": train " + format(r2_train, ".4f")
                  + " >> test " + format(summary["r2"], ".4f")
                  + " (gap " + format(gap, ".4f") + ")")

    # --- หาโมเดลที่ดีสุดของแต่ละตัวชี้วัด ---
    best_r2 = results[0]
    best_mse = results[0]
    best_mae = results[0]

    for summary in results:
        if summary["r2"] > best_r2["r2"]:
            best_r2 = summary
        if summary["mse"] < best_mse["mse"]:
            best_mse = summary
        if summary["mae"] < best_mae["mae"]:
            best_mae = summary

    print("R2  ดีสุด (สูงสุด): " + best_r2["model"]
          + "  (" + format(best_r2["r2"], ".4f") + ")")
    print("MSE ดีสุด (ต่ำสุด): " + best_mse["model"]
          + "  (" + format(best_mse["mse"], ".4f") + ")")
    print("MAE ดีสุด (ต่ำสุด): " + best_mae["model"]
          + "  (" + format(best_mae["mae"], ".4f") + ")")

    return results


if __name__ == "__main__":
    # ครอบ error ที่เกิดจากข้อมูล/ไฟล์ เพื่อพิมพ์ข้อความไทยสั้น ๆ แทน traceback ยาว
    try:
        dataframe = load_csv(CSV_PATH)
        compare(dataframe, target=TARGET)
    except ValueError as error:
        print("[error] " + str(error))
        sys.exit(1)
