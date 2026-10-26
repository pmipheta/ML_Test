"""
รันทุกโมเดล classification แล้วเรียงผลตาม F1

วิธีใช้:
    from run_all import run_all
    results = run_all(X, y, feature_names)

results = list[dict] เรียงจาก f1 มากไปน้อย
"""

from decision_tree import DecisionTreeModel
from knn_classifier import KNNModel
from logistic_regression import LogisticRegressionModel
from naive_bayes import NaiveBayesModel
from svm import SVMModel
from random_forest import RandomForestModel
from gradient_boosting import GradientBoostingModel
from adaboost import AdaBoostModel
from mlp import MLPModel
from Ida import LDAModel


def run_all(X, y, feature_names=None, verbose=True):
    models = [
        DecisionTreeModel(),
        KNNModel(),
        LogisticRegressionModel(),
        NaiveBayesModel(),
        SVMModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        AdaBoostModel(),
        MLPModel(),
        LDAModel(),
    ]

    results = []
    for model in models:
        result = model.run(X, y, feature_names)
        results.append(result)
        if verbose:
            print("done:", result["model"])

    results.sort(key=lambda item: item["f1"], reverse=True)

    if verbose:
        print_table(results)
    return results


def print_table(results):
    print()
    print("%-22s %8s %9s %7s %7s %8s" %
          ("model", "accuracy", "precision", "recall", "f1", "roc_auc"))
    print("-" * 66)
    for item in results:
        print("%-22s %8.4f %9.4f %7.4f %7.4f %8.4f" %
              (item["model"], item["accuracy"], item["precision"],
               item["recall"], item["f1"], item["roc_auc"]))


if __name__ == "__main__":
    # ทดสอบด้วยข้อมูลสังเคราะห์ (ไม่ต้องมีไฟล์จริง)
    from sklearn.datasets import make_classification
    X_demo, y_demo = make_classification(n_samples=2000, n_features=10,
                                         n_informative=5, weights=[0.84, 0.16],
                                         random_state=0)
    run_all(X_demo, y_demo)