"""Score saved student-outcome models on a test CSV. Models are not retrained."""

import json

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from theme import INK, PAGE, apply_theme
from schema import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    TARGET,
    TEST_CSV,
    compute_metrics,
    round_metric,
)

METRIC_LABELS = [
    ("accuracy", "Accuracy"),
    ("auc", "AUC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mcc", "MCC"),
]

st.set_page_config(
    page_title="Student outcome classifier",
    layout="wide",
)
apply_theme()


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def load_pipeline(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])


def read_table(source):
    frame = pd.read_csv(source)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    if TARGET not in frame.columns and hasattr(source, "seek"):
        source.seek(0)
        frame = pd.read_csv(source, sep=";")
        frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    return frame


def missing_columns(frame):
    required = FEATURE_COLUMNS + [TARGET]
    return [column for column in required if column not in frame.columns]


def score_frame(pipeline, frame):
    features = frame.loc[:, FEATURE_COLUMNS]
    actual = frame[TARGET]
    predicted = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    class_index = {label: index for index, label in enumerate(pipeline.classes_)}
    ordered = np.column_stack(
        [probabilities[:, class_index[label]] for label in CLASS_LABELS]
    )
    metrics = compute_metrics(actual, predicted, ordered, CLASS_LABELS)
    return actual, predicted, metrics


def comparison_frame(payload):
    rows = []
    for name in MODEL_FILES:
        record = payload["models"][name]
        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": record["accuracy"],
                "AUC": record["auc"],
                "Precision": record["precision"],
                "Recall": record["recall"],
                "F1": record["f1"],
                "MCC": record["mcc"],
            }
        )
    return pd.DataFrame(rows)


def format_metric(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def draw_confusion_matrix(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    figure, axis = plt.subplots(figsize=(5.4, 4.2), facecolor=PAGE)
    axis.set_facecolor("#FFFDF9")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=LinearSegmentedColormap.from_list(
            "pastel", ["#F7F4EE", "#D7EBE4", "#B7D4CB"]
        ),
        cbar=False,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        linewidths=0.6,
        linecolor=PAGE,
        annot_kws={"color": INK, "size": 11},
        ax=axis,
    )
    axis.set_xlabel("Predicted", color=INK)
    axis.set_ylabel("Actual", color=INK)
    axis.set_title("Confusion matrix", color=INK)
    axis.tick_params(colors=INK)
    figure.tight_layout()
    return figure


def main():
    payload = load_metrics()
    split = payload["split"]

    st.title("Student outcome classifier")
    st.markdown(
        "Predict whether a student will **drop out**, stay **enrolled**, or **graduate**, "
        "using the UCI student-dropout records. Pick a saved model and score a test CSV."
    )
    st.caption(
        f"Official comparison below uses the stratified holdout "
        f"({split['train_rows']} train / {split['test_rows']} test, "
        f"random_state {split['random_state']}). "
        "Precision, recall, and F1 are weighted. AUC is weighted one-vs-rest."
    )

    st.subheader("Held-out test comparison")
    table = comparison_frame(payload)
    best_name = table.sort_values(["F1", "MCC"], ascending=False).iloc[0]["ML Model Name"]
    display_table = table.copy()
    for column in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"):
        display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")
    st.dataframe(display_table, hide_index=True, width="stretch")
    st.caption(f"Highest weighted F1 on this split: {best_name}.")

    st.subheader("Score a test file")
    with st.sidebar:
        st.header("Evaluation")
        model_name = st.selectbox("Model", list(MODEL_FILES))
        upload = st.file_uploader(
            "Test CSV (features and Target)",
            type=["csv"],
            help="Use a test-sized file. The bundled holdout is scored when no file is uploaded.",
        )

    using_holdout = upload is None
    source = TEST_CSV if using_holdout else upload
    source_label = "bundled test_data.csv" if using_holdout else upload.name

    try:
        frame = read_table(source)
    except Exception as error:
        st.error(f"Could not read that CSV: {error}")
        return

    absent = missing_columns(frame)
    if absent:
        st.error("This CSV is missing columns the saved pipeline expects: " + ", ".join(absent))
        with st.expander("Expected columns"):
            st.write(FEATURE_COLUMNS + [TARGET])
        return
    if frame.empty:
        st.error("The CSV has a header but no rows.")
        return

    extras = [column for column in frame.columns if column not in FEATURE_COLUMNS + [TARGET]]
    st.markdown(f"Scoring **{model_name}** on `{source_label}` ({len(frame)} rows).")
    if extras:
        st.caption("Ignored extra columns: " + ", ".join(extras))

    pipeline = load_pipeline(model_name)
    with st.spinner("Scoring the saved model"):
        actual, predicted, metrics = score_frame(pipeline, frame)

    metric_columns = st.columns(3)
    for index, (key, label) in enumerate(METRIC_LABELS):
        metric_columns[index % 3].metric(label, format_metric(metrics[key]))
    if metrics["auc"] is None:
        st.caption("AUC needs all three classes (Dropout, Enrolled, Graduate) in the Target column.")

    if using_holdout:
        saved = payload["models"][model_name]
        matches = all(round_metric(metrics[key]) == saved[key] for key, _ in METRIC_LABELS)
        if matches:
            st.caption("Live scores on test_data.csv match the saved comparison table.")
        else:
            st.warning("Live scores on test_data.csv do not match model/metrics.json.")

    report = classification_report(
        actual,
        predicted,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).T.loc[CLASS_LABELS, ["precision", "recall", "f1-score", "support"]]
    report_frame = report_frame.rename(columns={"f1-score": "f1"})
    report_frame["support"] = report_frame["support"].astype(int)
    for column in ("precision", "recall", "f1"):
        report_frame[column] = report_frame[column].map(lambda value: f"{float(value):.4f}")

    matrix_column, report_column = st.columns(2)
    with matrix_column:
        figure = draw_confusion_matrix(actual, predicted)
        st.pyplot(figure, clear_figure=True)
    with report_column:
        st.markdown("**Classification report**")
        st.dataframe(report_frame, width="stretch")

    with st.expander("Preview of the scored rows"):
        st.dataframe(frame.head(12), width="stretch")


main()
