"""
Neural Network (MLP) — ใช้ sklearn

    model = MLPModel()
    result = model.run(X, y, feature_names)
"""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from data_structure import prepare, split_data, fit_best, evaluate


class MLPModel:
    def __init__(self, test_size=0.2, seed=42, cv_folds=5):
        self.test_size = test_size
        self.seed = seed
        self.cv_folds = cv_folds

    def run(self, X, y, feature_names=None):
        X, y, names = prepare(X, y, feature_names)
        X_train, X_test, y_train, y_test = split_data(X, y, self.test_size, self.seed)

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=500, early_stopping=True,
                                  random_state=self.seed)),
        ])
        grid = {
            "mlp__hidden_layer_sizes": [(32,), (64, 32)],
            "mlp__alpha": [0.0001, 0.01],
        }
        model, best_params = fit_best(estimator, grid, X_train, y_train, self.cv_folds)

        result = {"model": "Neural Network (MLP)", "params": best_params, "model_obj": model}
        result.update(evaluate(model, X_test, y_test))
        return result