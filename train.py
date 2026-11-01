"""Train five classifiers on the UCI student-dropout dataset and save them."""

import json
import urllib.request
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from schema import (
    BINARY_COLUMNS,
    CLASS_LABELS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    NOMINAL_COLUMNS,
    NUMERIC_COLUMNS,
    RAW_DATA_PATH,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    UCI_CSV_URL,
    compute_metrics,
    round_metric,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(classifier):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor()),
            ("classifier", classifier),
        ]
    )


def ensure_raw_data():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DATA_PATH.exists():
        print(f"Downloading {UCI_CSV_URL}")
        urllib.request.urlretrieve(UCI_CSV_URL, RAW_DATA_PATH)


def load_raw_frame():
    ensure_raw_data()
    frame = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    missing = [column for column in FEATURE_COLUMNS + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw file is missing columns: {missing}")
    grouped = NOMINAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS
    if set(grouped) != set(FEATURE_COLUMNS) or len(grouped) != len(FEATURE_COLUMNS):
        raise ValueError("Feature groups do not match the raw column list.")
    return frame.loc[:, FEATURE_COLUMNS + [TARGET]].copy()


def save_split(features, target, path):
    export = features.copy()
    export[TARGET] = target.to_numpy()
    export.loc[:, FEATURE_COLUMNS + [TARGET]].to_csv(path, index=False)


def per_class_report(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        label: {
            "precision": round_metric(report[label]["precision"]),
            "recall": round_metric(report[label]["recall"]),
            "f1": round_metric(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in CLASS_LABELS
    }


def evaluate_pipeline(pipeline, features, target):
    predicted = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    class_index = {label: index for index, label in enumerate(pipeline.classes_)}
    ordered = np.column_stack(
        [probabilities[:, class_index[label]] for label in CLASS_LABELS]
    )
    metrics = {
        key: round_metric(value)
        for key, value in compute_metrics(
            target, predicted, ordered, CLASS_LABELS
        ).items()
    }
    matrix = confusion_matrix(target, predicted, labels=CLASS_LABELS).tolist()
    return metrics, per_class_report(target, predicted), matrix


def fit_searched(classifier, param_grid, features, target):
    search = GridSearchCV(
        estimator=make_pipeline(classifier),
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        n_jobs=1,
    )
    search.fit(features, target)
    return search.best_estimator_, {
        "chosen": search.best_params_,
        "train_cv_f1_weighted": round_metric(search.best_score_),
    }


def main():
    frame = load_raw_frame()
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET]
    print(f"Rows={len(frame)} features={len(FEATURE_COLUMNS)}")
    print(target.value_counts().to_string())

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    save_split(x_train, y_train, TRAIN_CSV)
    save_split(x_test, y_test, TEST_CSV)
    print(f"Train rows={len(x_train)} test rows={len(x_test)}")

    estimators = []

    logistic = make_pipeline(
        LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )
    print("Fitting logistic regression")
    logistic.fit(x_train, y_train)
    estimators.append(
        (
            "Logistic Regression",
            logistic,
            {
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": "balanced",
            },
        )
    )

    print("Selecting decision tree depth on the training split")
    tree, tree_params = fit_searched(
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"classifier__max_depth": [4, 6, 8, 12, None]},
        x_train,
        y_train,
    )
    print(tree_params)
    estimators.append(("Decision Tree", tree, tree_params))

    print("Selecting k on the training split")
    knn, knn_params = fit_searched(
        KNeighborsClassifier(),
        {"classifier__n_neighbors": [3, 5, 11, 21]},
        x_train,
        y_train,
    )
    print(knn_params)
    estimators.append(("KNN", knn, knn_params))

    equal_prior = np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS))
    naive_bayes = make_pipeline(GaussianNB(priors=equal_prior))
    print("Fitting Gaussian naive Bayes")
    naive_bayes.fit(x_train, y_train)
    estimators.append(
        (
            "Naive Bayes",
            naive_bayes,
            {"variant": "GaussianNB", "priors": "equal"},
        )
    )

    svm = make_pipeline(
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            cache_size=1000,
        )
    )
    print("Fitting RBF support vector machine (this step is the slow one)")
    svm.fit(x_train, y_train)
    estimators.append(
        (
            "Support Vector Machine",
            svm,
            {
                "kernel": "rbf",
                "probability": True,
                "class_weight": "balanced",
            },
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_report = {}
    for name, pipeline, params in estimators:
        metrics, class_report, matrix = evaluate_pipeline(pipeline, x_test, y_test)
        path = MODEL_DIR / MODEL_FILES[name]
        joblib.dump(pipeline, path, compress=3)
        model_report[name] = {
            "file": MODEL_FILES[name],
            "params": params,
            **metrics,
            "per_class": class_report,
            "confusion_matrix": matrix,
            "confusion_labels": CLASS_LABELS,
        }
        print(name, metrics)

    payload = {
        "dataset": "Predict Students' Dropout and Academic Success (UCI id 697)",
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "class_counts": {label: int(count) for label, count in Counter(target).items()},
        "metric_definitions": {
            "precision_recall_f1": "weighted",
            "auc": "weighted one-vs-rest",
            "mcc": "multiclass Matthews correlation coefficient",
        },
        "models": model_report,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verify_saved_models(x_test, y_test, model_report)
    print(f"Wrote {METRICS_PATH}")


def verify_saved_models(x_test, y_test, model_report):
    saved_test = pd.read_csv(TEST_CSV)
    if not saved_test[FEATURE_COLUMNS].reset_index(drop=True).equals(
        x_test[FEATURE_COLUMNS].reset_index(drop=True)
    ):
        raise RuntimeError("test_data.csv does not match the held-out features.")
    if list(saved_test[TARGET]) != list(y_test):
        raise RuntimeError("test_data.csv targets do not match the held-out split.")
    for name, expected in model_report.items():
        pipeline = joblib.load(MODEL_DIR / MODEL_FILES[name])
        metrics, _, _ = evaluate_pipeline(pipeline, saved_test[FEATURE_COLUMNS], saved_test[TARGET])
        for key in ("accuracy", "auc", "precision", "recall", "f1", "mcc"):
            if metrics[key] != expected[key]:
                raise RuntimeError(
                    f"{name} {key} reloaded as {metrics[key]} but saved {expected[key]}"
                )
    print("Reloaded models match metrics.json on test_data.csv")


if __name__ == "__main__":
    main()
