"""
Gradient Boosting — ใช้ sklearn

    model = GradientBoostingModel()
    result = model.run(X, y, feature_names)
"""

from sklearn.ensemble import GradientBoostingClassifier
from data_structure import prepare, split_data, fit_best, evaluate


class GradientBoostingModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = GradientBoostingClassifier(random_state=self.seed)
        grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        }
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "Gradient Boosting", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result