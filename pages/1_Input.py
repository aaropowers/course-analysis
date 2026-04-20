from __future__ import annotations

import pandas as pd
import streamlit as st

from src.cleaning import build_transcript_from_course_list, parse_transcript
from src.utils import load_catalog, load_transcripts, normalize_course_number


st.title("Input")
st.caption("Load one of the prepared demo students, upload a simple CSV, or build a transcript manually.")

catalog_df = load_catalog()
demo_df = load_transcripts()

scenario_options = {
    "Early-stage student": "demo_early",
    "Mid-degree student": "demo_mid",
    "Upper-level student": "demo_upper",
}

with st.expander("Demo scenarios", expanded=True):
    scenario_name = st.selectbox("Select a demo profile", list(scenario_options))
    if st.button("Load demo transcript", type="primary"):
        student_id = scenario_options[scenario_name]
        st.session_state["transcript_df"] = demo_df[demo_df["student_id"] == student_id].copy()
        st.session_state["active_student_id"] = student_id
        st.success(f"Loaded {scenario_name.lower()} profile.")

with st.expander("Upload transcript CSV", expanded=True):
    st.caption("Expected columns: `course_number`, optional `grade`, optional `term`.")
    uploaded_file = st.file_uploader("Upload transcript", type=["csv"])
    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        parsed_df, invalid_courses = parse_transcript(upload_df)
        st.session_state["transcript_df"] = parsed_df
        st.session_state["active_student_id"] = "uploaded_student"
        st.session_state["invalid_courses"] = invalid_courses
        st.success(f"Loaded {len(parsed_df)} valid completed courses from upload.")
        if invalid_courses:
            st.warning("Ignored unknown courses: " + ", ".join(invalid_courses))

with st.expander("Manual transcript builder", expanded=False):
    selected_courses = st.multiselect(
        "Completed courses",
        options=catalog_df["course_number"].tolist(),
        default=st.session_state.get("transcript_df", pd.DataFrame()).get("course_number", pd.Series(dtype=str)).tolist(),
        format_func=lambda course: f"{course} - {catalog_df.loc[catalog_df['course_number'] == course, 'course_title'].iloc[0]}",
    )
    if st.button("Use selected courses"):
        st.session_state["transcript_df"] = build_transcript_from_course_list(selected_courses)
        st.session_state["active_student_id"] = "manual_student"
        st.success(f"Saved {len(selected_courses)} completed courses.")

st.subheader("Preferences")
interest_options = ["controls", "robotics", "thermal", "manufacturing", "data", "materials", "design", "energy"]
st.session_state["selected_interests"] = st.multiselect(
    "Interest areas",
    options=interest_options,
    default=st.session_state.get("selected_interests", ["controls", "robotics"]),
)
st.session_state["target_credit_load"] = st.slider(
    "Target credit load for next term",
    min_value=6,
    max_value=18,
    step=1,
    value=int(st.session_state.get("target_credit_load", 12)),
)

current_transcript = st.session_state.get("transcript_df", pd.DataFrame()).copy()
if not current_transcript.empty:
    current_transcript["course_number"] = current_transcript["course_number"].map(normalize_course_number)
    preview = current_transcript[["course_number", "grade", "term"]].fillna("")
    st.subheader("Current transcript in session")
    st.dataframe(preview, use_container_width=True, hide_index=True)
