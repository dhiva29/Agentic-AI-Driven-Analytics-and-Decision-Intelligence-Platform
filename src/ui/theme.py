from __future__ import annotations

import streamlit as st


def apply_theme(dark_mode: bool) -> None:
    if dark_mode:
        css = """
        <style>
        .stApp { background-color: #0f172a; color: #e2e8f0; }
        [data-testid="stSidebar"] { background: #111827; }
        .block-container { padding-top: 1.2rem; }
        </style>
        """
    else:
        css = """
        <style>
        .stApp { background-color: #f8fafc; color: #0f172a; }
        .block-container { padding-top: 1.2rem; }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)
