from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.audit import get_eligible_courses
from src.db import log_recommendations
from src.recommender import recommend_courses


def format_stat(value, fmt: str = ".2f", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def build_evidence_line(row) -> str:
    parts = [
        f"Avg GPA: {format_stat(row.evidence_avg_gpa)}",
        f"Pass rate: {format_stat(row.evidence_pass_rate, '.1f', '%')}",
        f"Records: {int(row.evidence_records) if not pd.isna(row.evidence_records) else 'N/A'}",
    ]
    if getattr(row, "evidence_top_companion", None):
        parts.append(f"Common with: {row.evidence_top_companion}")
    if getattr(row, "evidence_difficulty", None):
        parts.append(f"Load: {row.evidence_difficulty}")
    return " | ".join(parts)


def build_sequence_line(row) -> str | None:
    if getattr(row, "evidence_anchor_course", None) and not pd.isna(row.evidence_anchor_course):
        share = format_stat(row.evidence_share_after_anchor, ".1f", "%")
        count = int(row.evidence_after_anchor_count) if not pd.isna(row.evidence_after_anchor_count) else 0
        return f"Sequence evidence: often appears after {row.evidence_anchor_course} ({count} matching histories, {share} of anchor-to-target records)."
    return None


st.title("Recommendations")

transcript_df = st.session_state.get("transcript_df", pd.DataFrame())
if transcript_df.empty:
    st.warning("Load a transcript on the Input page to generate recommendations.")
    st.stop()

student_profile = {
    "student_id": st.session_state.get("active_student_id", "session_student"),
    "transcript_df": transcript_df,
    "interests": st.session_state.get("selected_interests", []),
    "target_credit_load": st.session_state.get("target_credit_load", 12),
}

eligible_df = get_eligible_courses(transcript_df)
recommendations = recommend_courses(student_profile, eligible_courses=eligible_df)
st.session_state["recommendations_df"] = recommendations
log_recommendations(student_profile["student_id"], recommendations.head(5))

top_n = min(8, len(recommendations))
st.subheader("Top next-course recommendations")
st.dataframe(
    recommendations[
        [
            "course_number",
            "course_title",
            "credits",
            "eligibility_status",
            "missing_coreqs",
            "score",
            "predicted_gpa",
            "evidence_avg_gpa",
            "evidence_pass_rate",
            "evidence_records",
        ]
    ].head(top_n),
    use_container_width=True,
    hide_index=True,
)

score_fig = px.bar(
    recommendations.head(top_n),
    x="course_number",
    y="score",
    color="requirement_priority",
    hover_data=["course_title", "predicted_gpa"],
    title="Recommendation score by course",
)
st.plotly_chart(score_fig, use_container_width=True)

st.subheader("Why these courses?")
for row in recommendations.head(6).itertuples():
    with st.container(border=True):
        st.markdown(f"**{row.course_number} - {row.course_title}**")
        st.write(row.explanation)
        st.caption(build_evidence_line(row))
        sequence_line = build_sequence_line(row)
        if sequence_line:
            st.caption(sequence_line)
        signal_parts = [
            f"Requirement priority: {row.requirement_priority:.2f}",
            f"Semester fit: {row.semester_fit:.2f}",
            f"Similarity: {row.similarity:.2f}",
            f"Co-enrollment: {row.coenrollment:.2f}",
            f"Collaborative: {row.collaborative:.2f}",
            f"Interest match: {row.interest_match:.2f}",
            f"Corequisite flag: {row.has_corequisite:.2f}",
        ]
        st.caption(f"Model signals: {' | '.join(signal_parts)}")
        if row.missing_coreqs:
            st.caption(f"Needs same-term corequisite(s): {row.missing_coreqs}")
