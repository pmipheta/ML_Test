"""
Multiple Linear Regression (หลาย features) — คำนวณเท่านั้น ไม่พิมพ์อะไร

วิธีใช้ (ผ่าน my_run.py):
    model = MultipleLinearRegression()
    result = model.run(X, y, feature_names)

สูตรที่ใช้:
    normal equation : w = (XᵀX)⁻¹ Xᵀy
    gradient descent: standardize X ก่อน แล้วไล่ปรับ w ทีละก้าว
"""

import numpy as np


class MultipleLinearRegression:
    def __init__(self, learning_rate=0.05, n_iterations=5000,
                 test_size=0.2, seed=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.seed = seed
        self.names = None

    # ------------------------------------------------------------------ data
    def set_data(self, X, y, feature_names=None):
        """
        เก็บข้อมูลลง instance (ใช้ได้กับ feature กี่ตัวก็ได้)

        Input:
            X: np.ndarray รูปร่าง (n, p)
            y: np.ndarray รูปร่าง (n,)
            feature_names: list[str] ชื่อคอลัมน์
        Output:
            ไม่คืนค่า แต่ตั้ง self.X (n, p), self.y (n,), self.names
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X = X
        self.y = np.asarray(y, dtype=float)

        names = []
        if feature_names is not None:
            for name in feature_names:
                names.append(name)
        else:
            for index in range(X.shape[1]):
                names.append("x" + str(index))
        self.names = names

    def split(self):
        """
        แบ่ง train/test แบบสุ่ม (seed คงที่ -> ทำซ้ำได้ผลเดิม)

        Input:
            ใช้ self.X, self.y, self.test_size, self.seed
        Output:
            ไม่คืนค่า แต่ตั้ง self.X_train, self.y_train, self.X_test, self.y_test
        """
        n = len(self.X)
        random_state = np.random.RandomState(self.seed)
        shuffled = random_state.permutation(n)
        n_train = int((1.0 - self.test_size) * n)

        train_idx = []
        test_idx = []
        position = 0
        while position < len(shuffled):
            if position < n_train:
                train_idx.append(shuffled[position])
            else:
                test_idx.append(shuffled[position])
            position += 1

        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]

    # ------------------------------------------------------------------- fit
    def build_design_matrix(self, X):
        """
        เติมคอลัมน์ 1 (intercept) ไว้หน้าสุด

        Input:
            X: np.ndarray รูปร่าง (n, p)
        Output:
            np.ndarray รูปร่าง (n, p+1)
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        ones = np.ones((n, 1))
        return np.hstack((ones, X))

    def fit_normal_equation(self, X=None, y=None):
        """
        แก้สมการปกติ  w = (XᵀX)⁻¹ Xᵀy

        Input:
            X: (n, p) — ถ้าไม่ส่งมาจะใช้ชุด train
            y: (n,)
        Output:
            w: (p+1,) โดย w[0] คือ intercept
        """
        if X is None:
            X = self.X_train
            y = self.y_train

        design = self.build_design_matrix(X)
        XtX = design.T @ design
        Xty = design.T @ y

        # ข้อมูลจริงมักมีสเกลต่างกันมาก (เช่น 1500 กับ 100000)
        # ทำให้ XᵀX แทบ singular -> inverse ได้ตัวเลขขยะ
        # จึงตรวจ cond ก่อน ถ้าไม่เสถียรให้ใช้ lstsq (SVD บน X ตรง ๆ) แทน
        with np.errstate(all="ignore"):
            condition = float(np.linalg.cond(XtX))

        if not np.isfinite(condition):
            weights = np.linalg.lstsq(design, y, rcond=None)[0]
        elif condition > 1e8:
            weights = np.linalg.lstsq(design, y, rcond=None)[0]
        else:
            XtX_inverse = np.linalg.inv(XtX)
            weights = XtX_inverse @ Xty

        return weights

    def fit_gradient_descent(self):
        """
        หาค่า w ด้วย gradient descent พร้อม standardization

        ขั้นตอน:
            1) scale: X_scaled = (X - mean) / std
            2) ไล่ปรับ w ทีละก้าวด้วย learning_rate
            3) แปลง w กลับเป็นหน่วยเดิม:
                   coef      = w / std
                   intercept = b - Σ(coef * mean)

        Input:
            ใช้ self.X_train, self.y_train, self.learning_rate, self.n_iterations
        Output:
            w: (p+1,) หน่วยเดิม (w[0] = intercept)
        """
        X = self.X_train
        y = self.y_train

        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        # กัน std = 0 (คอลัมน์ค่าคงที่)
        fixed_stds = []
        for value in stds:
            if np.isclose(value, 0.0):
                fixed_stds.append(1.0)
            else:
                fixed_stds.append(value)
        stds = np.asarray(fixed_stds)

        X_scaled = (X - means) / stds

        n = X_scaled.shape[0]
        p = X_scaled.shape[1]

        weights = np.zeros(p)
        intercept = 0.0

        iteration = 0
        while iteration < self.n_iterations:
            prediction = X_scaled @ weights + intercept
            error = prediction - y

            gradient_weights = (2.0 / n) * (X_scaled.T @ error)
            gradient_intercept = (2.0 / n) * np.sum(error)

            weights = weights - self.learning_rate * gradient_weights
            intercept = intercept - self.learning_rate * gradient_intercept
            iteration += 1

        # แปลงกลับเป็นหน่วยเดิม
        coefficients = []
        index = 0
        while index < len(weights):
            coefficients.append(weights[index] / stds[index])
            index += 1
        coefficients = np.asarray(coefficients)

        intercept_original = intercept - np.sum(coefficients * means)

        result = np.concatenate(([intercept_original], coefficients))
        return result

    # --------------------------------------------------------------- predict
    def predict(self, X, weights):
        """
        ทำนายค่า y ด้วยสมการเชิงเส้น

        Input:
            X: (m, p)
            weights: (p+1,) โดย weights[0] คือ intercept
        Output:
            np.ndarray รูปร่าง (m,)
        """
        design = self.build_design_matrix(X)
        return design @ weights

    # --------------------------------------------------------------- metrics
    def metrics(self, y_true, y_pred):
        """
        คำนวณ R², MSE, MAE

        Input:
            y_true: (n,), y_pred: (n,)
        Output:
            dict {"r2": float, "mse": float, "mae": float}
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        ss_res = 0.0
        absolute_sum = 0.0
        index = 0
        while index < len(y_true):
            residual = y_true[index] - y_pred[index]
            ss_res = ss_res + residual ** 2
            absolute_sum = absolute_sum + abs(residual)
            index += 1

        y_mean = np.mean(y_true)
        ss_tot = 0.0
        index = 0
        while index < len(y_true):
            ss_tot = ss_tot + (y_true[index] - y_mean) ** 2
            index += 1

        n = len(y_true)
        if ss_tot > 0:
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = float("nan")

        return {
            "r2": float(r2),
            "mse": float(ss_res / n),
            "mae": float(absolute_sum / n),
        }

    # ------------------------------------------------------------------- run
    def run(self, X=None, y=None, feature_names=None):
        """
        ทำงานทั้งหมด: เก็บข้อมูล -> split -> fit -> วัดผลบน test set

        Input:
            X, y, feature_names: ข้อมูล (ถ้าไม่ส่งมาใช้ที่ set_data ไว้แล้ว)
        Output:
            dict {"model", "features", "coefficients", "r2", "mse", "mae"}
        """
        if X is not None:
            self.set_data(X, y, feature_names)

        self.split()
        weights = self.fit_normal_equation()

        prediction = self.predict(self.X_test, weights)
        scores = self.metrics(self.y_test, prediction)
        train_scores = self.metrics(self.y_train, self.predict(self.X_train, weights))

        feature_text = ", ".join(self.names)

        return {
            "model": "Multiple Linear Regression",
            "features": feature_text,
            "coefficients": weights,
            "r2": scores["r2"],
            "r2_train": train_scores["r2"],
            "mse": scores["mse"],
            "mae": scores["mae"],
        }
