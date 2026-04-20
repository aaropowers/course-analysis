from __future__ import annotations

import streamlit as st

from src.db import init_db
from src.utils import load_transcripts


st.set_page_config(
    page_title="ME Course Planner",
    page_icon="⚙️",
    layout="wide",
)

init_db()

if "transcript_df" not in st.session_state:
    demo_transcripts = load_transcripts()
    st.session_state["transcript_df"] = demo_transcripts[demo_transcripts["student_id"] == "demo_mid"].copy()
    st.session_state["active_student_id"] = "demo_mid"
    st.session_state["selected_interests"] = ["controls", "robotics"]
    st.session_state["target_credit_load"] = 12
    st.session_state["invalid_courses"] = []

st.title("Transcript-Aware ME Course Planner")
st.caption(
    "Prototype advising dashboard for a one-week build. It combines structured degree rules "
    "with notebook-inspired recommendation signals and synthetic demo transcript profiles."
)

st.markdown(
    """
    Use the pages in the sidebar to:

    1. load a transcript manually or from CSV
    2. review degree progress and locked requirements
    3. inspect explainable course recommendations
    4. build a balanced next-semester bundle
    """
)

st.info(
    "This demo uses a curated mechanical engineering requirement map and synthetic student records "
    "for recommender behavior. It is intended for prototype demonstrations, not official advising."
)
