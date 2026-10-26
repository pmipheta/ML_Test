"""
Gaussian Naive Bayes — ใช้ sklearn

    model = NaiveBayesModel()
    result = model.run(X, y, feature_names)
"""

from sklearn.naive_bayes import GaussianNB
from data_structure import prepare, split_data, fit_best, evaluate


class NaiveBayesModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = GaussianNB()
        grid = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "Naive Bayes", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result