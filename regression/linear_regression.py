"""
Simple Linear Regression (1 feature) — คำนวณเท่านั้น ไม่พิมพ์อะไร

วิธีใช้ (ผ่าน my_run.py):
    model = LinearRegression()
    result = model.run(X, y, feature_names)

    result = {"model": "...", "feature": "...", "r2": ..., "mse": ..., "mae": ...}

ถ้า X มีหลายคอลัมน์ คลาสนี้จะเลือกคอลัมน์ที่ |correlation| กับ y สูงสุด
เพราะ simple linear regression ใช้ feature ได้ทีละตัวเท่านั้น
"""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=10000,
                 test_size=0.2, seed=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.seed = seed
        self.names = None

    # ------------------------------------------------------------------ data
    def set_data(self, X, y, feature_names=None):
        """
        เก็บข้อมูลลง instance และเลือก 1 feature ที่สัมพันธ์กับ y มากที่สุด

        Input:
            X: np.ndarray รูปร่าง (n,) หรือ (n, p)
            y: np.ndarray รูปร่าง (n,)
            feature_names: list[str] ชื่อคอลัมน์
        Output:
            ไม่คืนค่า แต่ตั้ง self.X (n,), self.y (n,), self.names
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float)

        # เตรียมชื่อคอลัมน์ (ถ้าไม่มีชื่อก็ตั้งเป็น x0, x1, ...)
        names = []
        if feature_names is not None:
            for name in feature_names:
                names.append(name)
        else:
            for index in range(X.shape[1]):
                names.append("x" + str(index))

        # ถ้ามีหลายคอลัมน์ ให้เหลือคอลัมน์เดียว
        if X.shape[1] > 1:
            correlations = []
            column_index = 0
            while column_index < X.shape[1]:
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.corrcoef(X[:, column_index], y)[0, 1]
                if np.isfinite(r):
                    correlations.append(abs(r))
                else:
                    correlations.append(0.0)
                column_index += 1

            best = int(np.argmax(correlations))
            X = X[:, [best]]

            selected_names = []
            selected_names.append(names[best])
            names = selected_names

        self.X = X[:, 0]
        self.y = y
        self.names = names

    def split(self):
        """
        แบ่งข้อมูลเป็น train/test แบบสุ่ม (seed คงที่ -> ทำซ้ำได้ผลเดิม)
        ทุกโมเดลใช้ seed เดียวกัน จึงได้แถว train/test ชุดเดียวกัน

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
    def fit_closed_form(self):
        """
        หา slope/intercept ด้วยสูตรตรง (least squares)
            slope     = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
            intercept = ȳ - slope * x̄

        Input:
            ใช้ self.X_train, self.y_train
        Output:
            (slope, intercept) เป็น float
        """
        x = self.X_train
        y = self.y_train

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = 0.0
        denominator = 0.0
        index = 0
        while index < len(x):
            numerator = numerator + (x[index] - x_mean) * (y[index] - y_mean)
            denominator = denominator + (x[index] - x_mean) ** 2
            index += 1

        # กันหารศูนย์: x คงที่ทั้งคอลัมน์ -> เส้นแนวนอนที่ mean(y)
        if np.isclose(denominator, 0.0):
            return 0.0, float(y_mean)

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return float(slope), float(intercept)

    def fit_gradient_descent(self):
        """
        หา slope/intercept ด้วย gradient descent
        (standardize x ก่อน แล้วค่อยแปลงกลับเป็นหน่วยเดิม)

            x_scaled = (x - mean) / std

        Input:
            ใช้ self.X_train, self.y_train, self.learning_rate, self.n_iterations
        Output:
            (slope, intercept) หน่วยเดิม
        """
        x = self.X_train
        y = self.y_train

        x_mean = float(np.mean(x))
        x_std = float(np.std(x))
        if np.isclose(x_std, 0.0):
            x_std = 1.0

        x_scaled = (x - x_mean) / x_std

        slope = 0.0
        intercept = 0.0
        n = len(x_scaled)

        iteration = 0
        while iteration < self.n_iterations:
            prediction = slope * x_scaled + intercept
            error = prediction - y

            gradient_slope = (2.0 / n) * np.sum(error * x_scaled)
            gradient_intercept = (2.0 / n) * np.sum(error)

            slope = slope - self.learning_rate * gradient_slope
            intercept = intercept - self.learning_rate * gradient_intercept
            iteration += 1

        slope_original = slope / x_std
        intercept_original = intercept - slope_original * x_mean
        return float(slope_original), float(intercept_original)

    # --------------------------------------------------------------- predict
    def predict(self, X, weights):
        """
        ทำนายค่า y จากสมการเส้นตรง

        Input:
            X: np.ndarray รูปร่าง (m,)
            weights: คู่ (slope, intercept)
        Output:
            np.ndarray รูปร่าง (m,)
        """
        slope = weights[0]
        intercept = weights[1]
        return slope * np.asarray(X, dtype=float) + intercept

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
            dict {"model", "feature", "r2", "mse", "mae"}
        """
        if X is not None:
            self.set_data(X, y, feature_names)

        self.split()
        weights = self.fit_closed_form()

        prediction = self.predict(self.X_test, weights)
        scores = self.metrics(self.y_test, prediction)
        train_scores = self.metrics(self.y_train, self.predict(self.X_train, weights))

        return {
            "model": "Simple Linear Regression",
            "feature": self.names[0],
            "r2": scores["r2"],
            "r2_train": train_scores["r2"],
            "mse": scores["mse"],
            "mae": scores["mae"],
        }
