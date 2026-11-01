"""Column groups and metric definitions shared by training and the app."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "students.csv"
TRAIN_CSV = ROOT / "train_data.csv"
TEST_CSV = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"

UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/697/data.csv"

TARGET = "Target"
CLASS_LABELS = ["Dropout", "Enrolled", "Graduate"]

NOMINAL_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

BINARY_COLUMNS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

NUMERIC_COLUMNS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

FEATURE_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]


def compute_metrics(y_true, y_pred, y_proba, labels):
    """Weighted multiclass metrics. AUC is weighted one-vs-rest."""
    y_true = np.asarray(y_true)
    present = set(np.unique(y_true))
    auc = None
    if present == set(labels):
        auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=list(labels),
            )
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "precision": float(
            precision_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=list(labels)
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=list(labels)
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=list(labels)
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def round_metric(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)
