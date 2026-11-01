"""Separate page for the train and test rows used to build the models."""

import pandas as pd
import streamlit as st

from schema import CLASS_LABELS, TARGET, TEST_CSV, TRAIN_CSV
from theme import BAR, apply_theme

st.set_page_config(page_title="Train and test data", layout="wide")
apply_theme()
st.title("Train and test data")
st.markdown(
    "These are the stratified 80/20 rows (`random_state=42`) used to fit the five pipelines "
    "and to fill the comparison table. The test file is the one the main page scores when no CSV is uploaded."
)


@st.cache_data
def load_split(path):
    return pd.read_csv(path)


def show_split(path):
    frame = load_split(path)
    counts = frame[TARGET].value_counts().reindex(CLASS_LABELS)
    row_count, column_count, feature_count = st.columns(3)
    row_count.metric("Rows", f"{len(frame):,}")
    column_count.metric("Columns", int(frame.shape[1]))
    feature_count.metric("Predictors", int(frame.shape[1] - 1))

    summary = pd.DataFrame(
        {
            "Class": CLASS_LABELS,
            "Students": [int(counts[label]) for label in CLASS_LABELS],
        }
    )
    summary["Share"] = summary["Students"] / summary["Students"].sum()
    count_column, chart_column = st.columns([1, 2])
    with count_column:
        st.dataframe(
            summary.style.format({"Share": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
    with chart_column:
        st.bar_chart(counts, sort=False, color=BAR)

    st.dataframe(frame, hide_index=True, width="stretch")


train_tab, test_tab = st.tabs(["Train", "Test"])
with train_tab:
    show_split(TRAIN_CSV)
with test_tab:
    show_split(TEST_CSV)
