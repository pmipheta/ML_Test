"""
ฟังก์ชันกลางที่ทุกโมเดล classification ใช้ร่วมกัน (ใช้ library sklearn)

- prepare   : แปลงข้อมูลเป็น numpy + ตั้งชื่อคอลัมน์
- split_data: แบ่ง train/test (seed คงที่ + stratify ให้สัดส่วนคลาสเท่าเดิม)
- fit_best  : เลือก hyperparameter ด้วย k-fold CV บน train เท่านั้น (กัน data leakage)
- evaluate  : วัดผลบน test set -> accuracy, precision, recall, f1, roc_auc
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


def prepare(X, y, feature_names=None):
    """
    Input:
        X: (n,) หรือ (n, p), y: (n,), feature_names: list[str]
    Output:
        X (n, p) float, y (n,) int, names list[str]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).astype(int)

    names = []
    if feature_names is not None:
        for name in feature_names:
            names.append(name)
    else:
        for index in range(X.shape[1]):
            names.append("x" + str(index))
    return X, y, names


def split_data(X, y, test_size=0.2, seed=42):
    """
    แบ่ง train/test ทุกโมเดลใช้ seed เดียวกัน จึงได้แถวชุดเดียวกัน
    Output: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


def fit_best(estimator, grid, X_train, y_train, cv_folds=5):
    """
    ลองทุกค่าใน grid ด้วย k-fold CV บน train แล้วคืนตัวที่ดีที่สุด
    ถ้า grid ว่าง = fit ตรง ๆ

    Output: (model ที่ fit แล้ว, dict ของ best params)
    """
    if not grid:
        estimator.fit(X_train, y_train)
        return estimator, {}

    if len(np.unique(y_train)) == 2:
        scoring = "f1"
    else:
        scoring = "f1_macro"

    search = GridSearchCV(estimator, grid, cv=cv_folds, scoring=scoring)
    search.fit(X_train, y_train)

    best_params = {}
    for key in search.best_params_:
        # ตัดคำนำหน้า "step__" ของ pipeline ออกให้อ่านง่าย
        short_key = key.split("__")[-1]
        best_params[short_key] = search.best_params_[key]
    return search.best_estimator_, best_params


def evaluate(model, X_test, y_test):
    """
    Output: dict {"accuracy", "precision", "recall", "f1", "roc_auc"}
    """
    prediction = model.predict(X_test)

    if len(np.unique(y_test)) == 2:
        average = "binary"
    else:
        average = "macro"

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average=average, zero_division=0)
    recall = recall_score(y_test, prediction, average=average, zero_division=0)
    f1 = f1_score(y_test, prediction, average=average, zero_division=0)

    # roc_auc ต้องใช้คะแนนความมั่นใจ ไม่ใช่คลาสที่ทำนาย
    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)
        else:
            score = model.decision_function(X_test)

        if average == "binary":
            if score.ndim == 2:
                score = score[:, 1]
            auc = roc_auc_score(y_test, score)
        else:
            auc = roc_auc_score(y_test, score, multi_class="ovr")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }