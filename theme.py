"""Shared light pastel styling so both pages stay readable."""

import streamlit as st

PAGE = "#F4F1EA"
PANEL = "#E7F2EF"
INK = "#1C2830"
LINE = "#D5E3DE"
BAR = "#9FC4BC"

_CSS = f"""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {PAGE};
    color: {INK};
}}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
    background-color: {PANEL};
    color: {INK};
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{
    color: {INK};
}}
h1, h2, h3, p, label, li,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {{
    color: {INK} !important;
}}
[data-testid="stMetric"] {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 0.55rem 0.75rem;
}}
code {{
    color: {INK} !important;
    background-color: #E3EFEA !important;
}}
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)
