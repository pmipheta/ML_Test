"""
Logistic Regression — ใช้ sklearn

    model = LogisticRegressionModel()
    result = model.run(X, y, feature_names)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_structure import prepare, split_data, fit_best, evaluate


class LogisticRegressionModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ])
        grid = {"lr__C": [0.01, 0.1, 1, 10]}
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "Logistic Regression", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result