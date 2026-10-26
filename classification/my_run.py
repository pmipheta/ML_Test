"""
ไฟล์เดียวที่ต้องแก้และรัน — ทำ 3 อย่าง:
    1) อ่าน DataFrame ของอาจารย์ (จาก CSV หรือส่ง df เข้ามา)
    2) ทำความสะอาดข้อมูลให้พร้อมใช้ (NaN / ข้อความ / วันที่ / ตัวเลขที่เก็บเป็นข้อความ)
    3) รันโมเดล classification ที่มีไฟล์อยู่ แล้วพิมพ์ตาราง Accuracy / Precision / Recall / F1
       + บอกโมเดลที่ดีสุดในแต่ละด้าน + เทียบกับ baseline

วิธีใช้:
    1) วางไฟล์ CSV ไว้โฟลเดอร์เดียวกับไฟล์นี้ (หรือใส่ path เต็ม)
    2) แก้ 2 บรรทัดในบล็อก config: CSV_PATH และ TARGET
    3) รัน:  python my_run.py

ถ้าต้องส่งอาจารย์แค่โมเดลเดียว:
    - เก็บไว้แค่ my_run.py + ไฟล์โมเดลตัวที่ต้องการ + data.csv
    - ตั้ง MODELS = ["logistic"] (หรือชื่อไฟล์ที่เหลือ)
    โปรแกรมจะรันเฉพาะโมเดลที่มีไฟล์อยู่ ไม่พังถ้าไฟล์อื่นหาย

ใน Jupyter:
    import pandas as pd
    from my_run import compare
    compare(pd.read_csv("data.csv"), target="approved")
"""

import sys

import numpy as np
import pandas as pd

# --- โหลดโมเดลเท่าที่มีในโฟลเดอร์ ---
# ถ้าไฟล์ไหนไม่มี ระบบจะข้ามโมเดลนั้นไป ไม่ทำให้ทั้งโปรแกรมพัง
try:
    from logistic_regression import LogisticRegressionModel
except ImportError:
    LogisticRegressionModel = None

try:
    from knn_classifier import KNNModel
    KNNClassifier = KNNModel
except ImportError:
    KNNClassifier = None

try:
    from decision_tree import DecisionTreeModel
    DecisionTreeClassifierModel = DecisionTreeModel
except ImportError:
    DecisionTreeClassifierModel = None

try:
    from naive_bayes import NaiveBayesModel
except ImportError:
    NaiveBayesModel = None

try:
    from svm import SVMModel
except ImportError:
    SVMModel = None

try:
    from random_forest import RandomForestModel
except ImportError:
    RandomForestModel = None

try:
    from gradient_boosting import GradientBoostingModel
except ImportError:
    GradientBoostingModel = None

try:
    from adaboost import AdaBoostModel
except ImportError:
    AdaBoostModel = None

try:
    from mlp import MLPModel
except ImportError:
    MLPModel = None

try:
    from Ida import LDAModel
except ImportError:
    LDAModel = None

# ===================== แก้ตรงนี้ =====================
CSV_PATH = "data.csv"
TARGET = "default"

# ถ้าส่งอาจารย์แค่บางโมเดล ให้ใส่เฉพาะชื่อที่มี เช่น MODELS = ["knn"]
MODELS = ["logistic", "knn", "tree", "nb", "svm", "rf", "gb", "ada", "mlp", "lda"]
DROP_COLS = ["app_id"]      # คอลัมน์ ID/ชื่อ/วันที่ ที่ไม่ควรใช้เป็น feature
# ====================================================

TEST_SIZE = 0.2             # เก็บ 20% ไว้ทดสอบ
SEED = 42                   # seed เดียวกันทุกโมเดล (เทียบกันยุติธรรม)
MAX_TEXT_CATEGORIES = 15    # คอลัมน์ข้อความที่มีหมวดเกินนี้จะถูกข้าม
OUTLIER_FACTOR = 1.5        # ตัวคูณ IQR (1.5 = มาตรฐาน, 3.0 = ตัดเฉพาะค่าสุดโต่งมาก)
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
    ซึ่งซ้ำกับ intercept (และทำให้ข้อมูลซ้ำซ้อน) โมเดลจึงเรียนรู้ยากขึ้น

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
# 3) หา outlier ด้วยกฎ IQR
# ---------------------------------------------------------------------------
def inlier_mask(values, factor=OUTLIER_FACTOR):
    """
    คืน mask ว่าแถวไหน "อยู่ในช่วงปกติ" ตามกฎ IQR

    กฎ IQR (มาตรฐานของงานสถิติ):
        Q1  = ควอไทล์ที่ 25%
        Q3  = ควอไทล์ที่ 75%
        IQR = Q3 - Q1
        ค่าปกติ = อยู่ในช่วง [Q1 - factor*IQR , Q3 + factor*IQR]

    ถ้า IQR = 0 (ค่าส่วนใหญ่ซ้ำกัน) จะตัดสินไม่ได้ จึงไม่ตัดแถวไหนเลย

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
        return np.ones(len(values), dtype=bool)

    low = q1 - factor * iqr
    high = q3 + factor * iqr

    return (values >= low) & (values <= high)


# ---------------------------------------------------------------------------
# 3.5) แปลง target ให้เป็น label 0..k-1
# ---------------------------------------------------------------------------
def encode_target(series):
    """
    รับ target ที่เป็นข้อความหรือตัวเลข แล้วคืนค่าที่พร้อมใช้

    - ถ้า target เป็นตัวเลข (หรือข้อความที่แปลงเป็นตัวเลขได้ >= 90%)
      จะเรียงค่าตัวเลขจากน้อยไปมาก
    - ถ้าเป็นข้อความ จะเรียงชื่อคลาสตามตัวอักษร
    การเรียงทำให้ได้ผลเหมือนเดิมทุกครั้งที่รัน (reproducible)

    Input:
        series: pandas.Series ของ target
    Output:
        (ค่าที่ normalize แล้ว, class_names list, numeric_mode bool)
    """
    valid_mask = series.notna()
    non_null_count = int(valid_mask.sum())

    numeric_mode = False
    normalized = None

    if pd.api.types.is_numeric_dtype(series):
        normalized = pd.to_numeric(series, errors="coerce")
        numeric_mode = True
    else:
        cleaned = series.astype("string")
        cleaned = cleaned.str.replace(",", "", regex=False)
        cleaned = cleaned.str.strip()
        as_number = pd.to_numeric(cleaned, errors="coerce")
        number_count = int(as_number.notna().sum())

        if non_null_count > 0 and number_count >= 0.9 * non_null_count:
            normalized = as_number
            numeric_mode = True
        else:
            normalized = series.astype("string")
            numeric_mode = False

    classes = []

    if numeric_mode:
        unique_values = pd.unique(normalized.dropna())
        sorted_values = np.sort(unique_values)
        index = 0
        while index < len(sorted_values):
            classes.append(sorted_values[index])
            index += 1
    else:
        unique_values = pd.unique(normalized[valid_mask])
        sorted_values = sorted(unique_values)
        index = 0
        while index < len(sorted_values):
            classes.append(str(sorted_values[index]))
            index += 1

    return normalized, classes, numeric_mode


# ---------------------------------------------------------------------------
# 4) ทำความสะอาดข้อมูล -> numpy arrays
# ---------------------------------------------------------------------------
def prepare_data(df, target, features=None):
    """
    แปลง DataFrame เป็น X, y พร้อมจัดการปัญหาที่เจอได้จริง

    สิ่งที่ทำให้อัตโนมัติ:
        - target ข้อความ -> label 0..k-1 (เรียงชื่อคลาสเพื่อให้ซ้ำได้)
        - ตัดแถวที่มี NaN หรือ inf
        - ตัดคอลัมน์ค่าคงที่ และคอลัมน์ที่ว่างทั้งคอลัมน์
        - ข้อความ -> one-hot ; ตัวเลขที่เก็บเป็นข้อความ ("1500", "51,598") -> ตัวเลข
        - คอลัมน์วันที่ และข้อความที่มีหมวดเยอะเกินไป -> ข้าม
        - ตัด outlier ด้วย IQR ถ้าตัดไม่เกิน OUTLIER_MAX_FRACTION

    Input:
        df: pandas.DataFrame
        target: str ชื่อคอลัมน์เป้าหมาย (ข้อความหรือตัวเลขก็ได้)
        features: None = ใช้ทุกคอลัมน์ที่เหลือ, หรือ list[str] เลือกเอง
    Output:
        X (n, p), y (n,), feature_names list[str], class_names list, info dict
    """
    if df is None:
        raise ValueError("ไม่ได้ส่ง DataFrame มา")

    work = df.copy()
    n_original = len(work)
    duplicate_rows = int(work.duplicated().sum())
    # ตัดแถวซ้ำทั้งแถวออก (กัน train/test contamination)
    work = work.drop_duplicates().reset_index(drop=True)

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

    # --- แปลง target เป็น label จำนวนเต็ม 0..k-1 ---
    normalized_target, class_names, numeric_mode = encode_target(work[target])

    if len(class_names) < 2:
        raise ValueError(
            "คอลัมน์ target " + repr(target) + " มีคลาสน้อยกว่า 2 คลาส"
            " — งาน classification ต้องมีอย่างน้อย 2 คลาส"
        )

    y_encoded = pd.Series(np.nan, index=work.index, dtype=float)
    class_index = 0
    while class_index < len(class_names):
        class_value = class_names[class_index]
        if numeric_mode:
            match_mask = (normalized_target == class_value)
        else:
            match_mask = (normalized_target == str(class_value))
        match_mask = match_mask.fillna(False)
        y_encoded[match_mask] = float(class_index)
        class_index += 1

    # --- เลือกคอลัมน์ที่จะใช้เป็น feature ---
    if features is None:
        feature_columns = []
        for column in work.columns:
            if column == target:
                continue
            if column in DROP_COLS:
                continue
            feature_columns.append(column)
    else:
        selected = features
        if isinstance(features, str):
            selected = [features]

        feature_columns = []
        for column in selected:
            if column == target:
                continue
            if column in DROP_COLS:
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

    # --- วนทีละคอลัมน์: ตัวเลขใช้ตรง ๆ / ข้อความแปลงเป็น 0-1 ---
    part_series = []
    part_names = []

    for column in feature_columns:
        series = work[column]

        # 0) คอลัมน์ index ที่ pandas สร้างตอน save CSV (Unnamed: 0) -> ข้าม
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
        encoded_parts = encode_text_column(series, str(column), MAX_TEXT_CATEGORIES)[0]
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
    # isfinite ครอบทั้ง NaN และ inf และทำทีเดียวทั้งเมทริกซ์ (ไม่วนทีละแถว)
    X_values = X_frame.to_numpy(dtype=float)
    y_values = y_encoded.to_numpy(dtype=float)

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

    # --- ตัด outlier ด้วยกฎ IQR (เฉพาะ feature) ---
    # แถวที่สุดโต่งในคอลัมน์ใดคอลัมน์หนึ่งจะถูกตัดทั้งแถว
    # แต่ถ้าต้องตัดเกิน OUTLIER_MAX_FRACTION แปลว่าข้อมูลกระจายกว้างจริง -> ไม่ตัด
    #
    # หมายเหตุ: ไม่ตัด outlier ที่ target เพราะงาน classification target คือ "คลาส"
    # ไม่ใช่ค่าต่อเนื่อง ถ้าเอา IQR ไปใช้จะตัดแถวของคลาสที่พบน้อย (คลาสเกิดยาก)
    # ทิ้งไปทั้งคลาส ซึ่งทำลายงานจำแนกประเภท
    outlier_keep = np.ones(X_values.shape[0], dtype=bool)

    column_index = 0
    while column_index < X_values.shape[1]:
        outlier_keep = outlier_keep & inlier_mask(X_values[:, column_index])
        column_index += 1

    flagged = int((~outlier_keep).sum())
    outliers_removed = 0
    outliers_skipped = 0

    if flagged > 0:
        fraction = flagged / X_values.shape[0]
        if fraction <= OUTLIER_MAX_FRACTION:
            X_values = X_values[outlier_keep]
            y_values = y_values[outlier_keep]
            outliers_removed = flagged
        else:
            outliers_skipped = flagged

    # --- ตรวจจำนวนแถวและจำนวนคลาสที่เหลือ ---
    if X_values.shape[0] < 10:
        raise ValueError(
            "เหลือข้อมูลใช้ได้ " + str(X_values.shape[0]) + " แถว (ต้องมีอย่างน้อย 10)"
        )

    remaining_labels = np.unique(y_values)
    if len(remaining_labels) < 2:
        raise ValueError(
            "หลังทำความสะอาดแล้ว target " + repr(target)
            + " เหลือคลาสเดียว ทำนายไม่ได้"
        )

    y_values = y_values.astype(int)

    info = {}
    info["rows_original"] = n_original
    info["rows_used"] = int(X_values.shape[0])
    info["duplicate_rows"] = duplicate_rows
    info["outliers_removed"] = outliers_removed
    info["outliers_skipped"] = outliers_skipped

    return X_values, y_values, final_names, class_names, info


# ---------------------------------------------------------------------------
# 5) รันทุกโมเดลแล้วเทียบ
# ---------------------------------------------------------------------------
def compare(df, target, features=None, test_size=TEST_SIZE, seed=SEED):
    """
    รันโมเดล classification ที่มีไฟล์อยู่บนข้อมูลชุดเดียวกัน แล้วพิมพ์ตาราง

    Input:
        df: pandas.DataFrame
        target: str ชื่อคอลัมน์เป้าหมาย
        features: None หรือ list[str]
        test_size: float สัดส่วนข้อมูลทดสอบ
        seed: int seed สำหรับแบ่ง train/test (ใช้ร่วมกันทุกโมเดล)
    Output:
        list ของ dict ผลแต่ละโมเดล
    """
    X, y, feature_names, class_names, info = prepare_data(df, target, features)

    # --- ตรวจชื่อโมเดลใน MODELS ---
    known_names = ["logistic", "knn", "tree", "nb", "svm", "rf", "gb", "ada", "mlp", "lda"]
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
    model_classes.append(("logistic", LogisticRegressionModel))
    model_classes.append(("knn", KNNClassifier))
    model_classes.append(("tree", DecisionTreeClassifierModel))
    model_classes.append(("nb", NaiveBayesModel))
    model_classes.append(("svm", SVMModel))
    model_classes.append(("rf", RandomForestModel))
    model_classes.append(("gb", GradientBoostingModel))
    model_classes.append(("ada", AdaBoostModel))
    model_classes.append(("mlp", MLPModel))
    model_classes.append(("lda", LDAModel))

    # --- เตรียมรายชื่อโมเดลที่จะรัน (ข้ามตัวที่ไม่มีไฟล์) ---
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

    for message in skipped:
        print("[skip] " + message)

    if len(models) == 0:
        raise ValueError(
            "ไม่มีโมเดลให้รันเลย — ตรวจสอบว่ามีไฟล์โมเดลอยู่ในโฟลเดอร์ "
            "และชื่อใน MODELS ถูกต้อง"
        )

    # --- รันทีละโมเดลด้วย test_size/seed เดียวกัน ---
    results = []
    ran_models = []

    for model in models:
        model.test_size = test_size
        model.seed = seed

        try:
            summary = model.run(X, y, feature_names)
        except ValueError as error:
            message = model.__class__.__name__ + " (" + str(error) + ")"
            skipped.append(message)
            print("[skip] " + message)
            continue

        results.append(summary)
        ran_models.append(model)

    if len(results) == 0:
        raise ValueError("รันทุกโมเดลไม่สำเร็จ: " + str(skipped))

    # --- พิมพ์หัวตาราง ---
    header = "rows " + str(info["rows_used"]) + "/" + str(info["rows_original"])
    header = header + " | features " + str(len(feature_names))
    header = header + " | target " + str(target)
    header = header + " | classes " + str(len(class_names))
    if info.get("duplicate_rows", 0) > 0:
        header = header + " | duplicates " + str(info["duplicate_rows"])
    if info.get("outliers_removed", 0) > 0:
        header = header + " | outliers " + str(info["outliers_removed"])
    if info.get("outliers_skipped", 0) > 0:
        header = header + " | outliers skipped " + str(info["outliers_skipped"])
    print(header)
    print()

    table_header = "Model".ljust(34)
    table_header = table_header + "Accuracy".rjust(12)
    table_header = table_header + "Precision".rjust(12)
    table_header = table_header + "Recall".rjust(12)
    table_header = table_header + "F1".rjust(12)
    print(table_header)
    print("-" * 82)

    for summary in results:
        line = summary["model"].ljust(34)
        line = line + format(summary["accuracy"], ">12.4f")
        line = line + format(summary["precision"], ">12.4f")
        line = line + format(summary["recall"], ">12.4f")
        line = line + format(summary["f1"], ">12.4f")
        print(line)

    print("-" * 82)

    # --- หาโมเดลที่ดีสุดของแต่ละตัวชี้วัด (ค่าสูงสุดชนะทุกตัว) ---
    best_accuracy = results[0]
    best_precision = results[0]
    best_recall = results[0]
    best_f1 = results[0]

    for summary in results:
        if summary["accuracy"] > best_accuracy["accuracy"]:
            best_accuracy = summary
        if summary["precision"] > best_precision["precision"]:
            best_precision = summary
        if summary["recall"] > best_recall["recall"]:
            best_recall = summary
        if summary["f1"] > best_f1["f1"]:
            best_f1 = summary

    print("Accuracy ดีสุด (สูงสุด): " + best_accuracy["model"]
          + "  (" + format(best_accuracy["accuracy"], ".4f") + ")")
    print("Precision ดีสุด (สูงสุด): " + best_precision["model"]
          + "  (" + format(best_precision["precision"], ".4f") + ")")
    print("Recall ดีสุด (สูงสุด): " + best_recall["model"]
          + "  (" + format(best_recall["recall"], ".4f") + ")")
    print("F1 ดีสุด (สูงสุด): " + best_f1["model"]
          + "  (" + format(best_f1["f1"], ".4f") + ")")

    # --- feature importance: พิมพ์ top features ของโมเดลที่รองรับ ---
    print()
    print("Feature importance (top 5):")
    for result in results:
        model = result.get("model_obj")
        if model is None:
            continue
        # unwrap pipeline
        estimator = model
        if hasattr(estimator, "named_steps"):
            # หา step สุดท้ายที่มี feature_importances_ หรือ coef_
            last = list(estimator.named_steps.values())[-1]
            estimator = last
        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.abs(coef)
        if importances is None:
            continue
        order = np.argsort(importances)[::-1][:5]
        pairs = []
        for idx in order:
            if idx < len(feature_names):
                pairs.append(feature_names[idx] + "=" + format(float(importances[idx]), ".4f"))
        print("  " + result["model"] + ": " + ", ".join(pairs))

    # --- baseline: ถ้าทำนายคลาสที่บ่อยที่สุดใน test set ทุกแถว จะได้ accuracy เท่าไร ---
    from sklearn.model_selection import train_test_split as _split
    _, _, _, _y_test = _split(X, y, test_size=test_size,
                              random_state=seed, stratify=y)
    labels, counts = np.unique(np.asarray(_y_test), return_counts=True)
    most_common_count = int(np.max(counts))
    baseline = most_common_count / len(_y_test)
    print("baseline (ทำนายคลาสที่บ่อยสุด): " + format(baseline, ".4f"))

    return results


if __name__ == "__main__":
    # ครอบ error ที่เกิดจากข้อมูล/ไฟล์ เพื่อพิมพ์ข้อความไทยสั้น ๆ แทน traceback ยาว
    try:
        dataframe = load_csv(CSV_PATH)
        compare(dataframe, target=TARGET)
    except ValueError as error:
        print("[error] " + str(error))
        sys.exit(1)
