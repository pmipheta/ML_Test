"""
Polynomial Regression (1 feature) — คำนวณเท่านั้น ไม่พิมพ์อะไร

วิธีใช้ (ผ่าน my_run.py):
    model = PolynomialRegression()
    result = model.run(X, y, feature_names)

แนวคิด: สร้าง feature ใหม่เป็น x, x², x³, ... แล้วใช้ least squares ธรรมดา
เลือก degree ที่ดีที่สุดด้วย k-fold cross-validation บนข้อมูล train เท่านั้น
(ไม่ใช้ test set เลือก เพื่อกัน data leakage)
"""

import numpy as np


class PolynomialRegression:
    def __init__(self, degrees=None, test_size=0.2, seed=42,
                 cv_folds=5, cv_seed=1234):
        if degrees is None:
            degrees = []
            for degree in range(1, 10):
                degrees.append(degree)
        self.degrees = list(degrees)
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds
        self.cv_seed = cv_seed
        self.names = None
        self.scalers = {}       # {degree: (mean, std)} ของ x_train
        self.cv_mse = []
        self.cv_se = []

    # ------------------------------------------------------------------ data
    def set_data(self, X, y, feature_names=None):
        """
        เก็บข้อมูล และเลือก 1 feature ที่สัมพันธ์กับ y มากที่สุด

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

        names = []
        if feature_names is not None:
            for name in feature_names:
                names.append(name)
        else:
            for index in range(X.shape[1]):
                names.append("x" + str(index))

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
        แบ่ง train/test แบบสุ่ม (seed คงที่)

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

    # -------------------------------------------------------------- features
    def make_poly_features(self, x, degree):
        """
        สร้าง design matrix [1, x, x², ..., x^degree]

        Input:
            x: np.ndarray รูปร่าง (n,)
            degree: int กำลังสูงสุด
        Output:
            np.ndarray รูปร่าง (n, degree+1)
        """
        x = np.asarray(x, dtype=float)

        columns = []
        ones = np.ones_like(x)
        columns.append(ones)

        power = 1
        while power <= degree:
            columns.append(x ** power)
            power += 1

        return np.column_stack(columns)

    def scale_x(self, x, degree, fit_scaler):
        """
        Standardize x ก่อนยกกำลัง (กันค่าล้นเมื่อ degree สูง)

        Input:
            x: np.ndarray รูปร่าง (n,)
            degree: int
            fit_scaler: True = คำนวณ mean/std ใหม่จาก x ที่ส่งมา (ใช้ตอน fit)
                        False = ใช้ mean/std ที่เก็บไว้ (ใช้ตอน predict)
        Output:
            np.ndarray รูปร่าง (n, degree+1)
        """
        x = np.asarray(x, dtype=float)

        if fit_scaler:
            x_mean = float(np.mean(x))
            x_std = float(np.std(x))
            if np.isclose(x_std, 0.0):
                x_std = 1.0
            self.scalers[degree] = (x_mean, x_std)
        else:
            if degree in self.scalers:
                x_mean = self.scalers[degree][0]
                x_std = self.scalers[degree][1]
            else:
                x_mean = float(np.mean(x))
                x_std = float(np.std(x))
                if np.isclose(x_std, 0.0):
                    x_std = 1.0

        x_scaled = (x - x_mean) / x_std
        return self.make_poly_features(x_scaled, degree)

    # ------------------------------------------------------------------- fit
    def fit(self, degree, X=None, y=None):
        """
        Fit polynomial ด้วย least squares (ใช้ lstsq เพื่อความเสถียร)

        Input:
            degree: int กำลังสูงสุด
            X, y: ข้อมูล (ถ้าไม่ส่งมาใช้ชุด train)
        Output:
            w: np.ndarray รูปร่าง (degree+1,)
        """
        if X is None:
            X = self.X_train
            y = self.y_train

        design = self.scale_x(X, degree, True)
        weights = np.linalg.lstsq(design, y, rcond=None)[0]
        return weights

    def predict(self, X, weights):
        """
        ทำนายค่า y จาก coefficient

        Input:
            X: np.ndarray รูปร่าง (m,)
            weights: np.ndarray รูปร่าง (degree+1,)
        Output:
            np.ndarray รูปร่าง (m,)
        """
        degree = len(weights) - 1
        design = self.scale_x(X, degree, False)
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

    # ------------------------------------------------------- เลือก degree
    def select_degree(self):
        """
        เลือก degree ด้วย k-fold cross-validation บน train เท่านั้น
        ใช้กฎ 1-SE: เลือก degree เล็กสุดที่ CV MSE ยังอยู่ใน min + 1 SE

        ขั้นตอน:
            1) สับ index ของ train แล้วแบ่งเป็น k ส่วน
            2) แต่ละ degree: ฝึกบน k-1 ส่วน วัดผลบนส่วนที่เหลือ ทำครบ k รอบ
            3) เอา MSE เฉลี่ยของแต่ละ degree มาเลือก

        Input:
            ใช้ self.X_train, self.y_train, self.degrees, self.cv_folds, self.cv_seed
        Output:
            int degree ที่เลือก (เก็บ self.cv_mse ไว้ด้วย)
        """
        X = self.X_train
        y = self.y_train
        n = len(X)

        k = self.cv_folds
        if k < 2:
            k = 2
        if k > n:
            k = n

        random_state = np.random.RandomState(self.cv_seed)
        shuffled = random_state.permutation(n)

        # แบ่ง index เป็น k กอง (กองสุดท้ายอาจเล็กกว่า)
        fold_size = n // k
        folds = []
        fold_index = 0
        while fold_index < k:
            start = fold_index * fold_size
            if fold_index == k - 1:
                end = n
            else:
                end = start + fold_size
            folds.append(shuffled[start:end])
            fold_index += 1

        self.cv_mse = []
        self.cv_se = []

        for degree in self.degrees:
            fold_scores = []
            validation_position = 0
            while validation_position < len(folds):
                validation_idx = folds[validation_position]

                # ข้อมูลฝึกในรอบนี้ = ทุกแถวที่ไม่ได้อยู่ใน fold ตรวจ
                # is_validation ชี้ด้วยเลขแถวจริง จึงแปลงกลับเป็นเลขแถวด้วย where
                # ห้ามใช้ shuffled[~is_validation] เพราะ shuffled เรียงตามตำแหน่ง
                # ไม่ใช่ตามเลขแถว จะได้ชุดแถวผิด (แถว fold ตรวจรั่วเข้าไปฝึก
                # และบางแถวหายไปจากทั้งฝึกและตรวจ)
                is_validation = np.zeros(n, dtype=bool)
                is_validation[validation_idx] = True
                fit_idx = np.where(~is_validation)[0]

                # scaler คำนวณจาก fold train เท่านั้น
                x_fit = X[fit_idx]
                x_val = X[validation_idx]
                fit_mean = float(np.mean(x_fit))
                fit_std = float(np.std(x_fit))
                if np.isclose(fit_std, 0.0):
                    fit_std = 1.0

                design_fit = self.make_poly_features((x_fit - fit_mean) / fit_std, degree)
                design_val = self.make_poly_features((x_val - fit_mean) / fit_std, degree)

                weights = np.linalg.lstsq(design_fit, y[fit_idx], rcond=None)[0]
                prediction = design_val @ weights

                squared_errors = (prediction - y[validation_idx]) ** 2
                fold_scores.append(float(np.mean(squared_errors)))

                validation_position += 1

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            standard_error = std_score / np.sqrt(len(fold_scores))

            self.cv_mse.append(mean_score)
            self.cv_se.append(standard_error)

        # หา degree ที่ CV ต่ำสุดก่อน
        best_index = int(np.argmin(self.cv_mse))

        # แล้วใช้กฎ 1-SE: เลือก degree เล็กสุดที่ยังอยู่ใน min + SE
        threshold = self.cv_mse[best_index] + self.cv_se[best_index]

        candidates = []
        index = 0
        while index < len(self.degrees):
            if self.cv_mse[index] <= threshold:
                candidates.append(index)
            index += 1

        chosen_index = candidates[0]
        for candidate in candidates:
            if self.degrees[candidate] < self.degrees[chosen_index]:
                chosen_index = candidate

        return int(self.degrees[chosen_index])

    # ------------------------------------------------------------------- run
    def run(self, X=None, y=None, feature_names=None):
        """
        ทำงานทั้งหมด: เก็บข้อมูล -> split -> เลือก degree -> fit -> วัดผล test

        Input:
            X, y, feature_names: ข้อมูล (ถ้าไม่ส่งมาใช้ที่ set_data ไว้แล้ว)
        Output:
            dict {"model", "degree", "feature", "r2", "mse", "mae"}
        """
        if X is not None:
            self.set_data(X, y, feature_names)

        self.split()
        degree = self.select_degree()

        weights = self.fit(degree)
        prediction = self.predict(self.X_test, weights)
        scores = self.metrics(self.y_test, prediction)
        train_scores = self.metrics(self.y_train, self.predict(self.X_train, weights))

        return {
            "model": "Polynomial Regression (degree " + str(degree) + ")",
            "degree": degree,
            "feature": self.names[0],
            "r2": scores["r2"],
            "r2_train": train_scores["r2"],
            "mse": scores["mse"],
            "mae": scores["mae"],
        }
