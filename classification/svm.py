"""
Support Vector Machine (RBF kernel) — ใช้ sklearn

    model = SVMModel()
    result = model.run(X, y, feature_names)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_structure import prepare, split_data, fit_best, evaluate


class SVMModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
        ])
        grid = {"svm__C": [0.1, 1, 10]}
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "SVM", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result