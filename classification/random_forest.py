"""
Random Forest — ใช้ sklearn

    model = RandomForestModel()
    result = model.run(X, y, feature_names)
"""

from sklearn.ensemble import RandomForestClassifier
from data_structure import prepare, split_data, fit_best, evaluate


class RandomForestModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = RandomForestClassifier(n_estimators=200,
                                           class_weight="balanced",
                                           random_state=self.seed,
                                           n_jobs=-1)
        grid = {
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 5],
        }
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "Random Forest", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result