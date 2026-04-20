from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.recommender import recommend_courses
from src.semester_planner import build_semester_plan


def format_stat(value, fmt: str = ".2f", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def build_plan_evidence_line(row) -> str:
    parts = [
        f"Avg GPA: {format_stat(row.evidence_avg_gpa)}",
        f"Pass rate: {format_stat(row.evidence_pass_rate, '.1f', '%')}",
        f"Records: {int(row.evidence_records) if not pd.isna(row.evidence_records) else 'N/A'}",
    ]
    if getattr(row, "bundle_top_partner", None):
        parts.append(f"Pairs with {row.bundle_top_partner} ({int(row.bundle_partner_count)} terms)")
    if getattr(row, "bundle_coreq_ready", None) == "Yes" and getattr(row, "missing_coreqs", ""):
        parts.append("Coreq included")
    return " | ".join(parts)


def build_bundle_rationale(row) -> str:
    notes = [f"{row.bundle_reason}."]
    if getattr(row, "bundle_top_partner", None):
        notes.append(
            f"Historically, students often bundled it with {row.bundle_top_partner} in {int(row.bundle_partner_count)} same-term records."
        )
    if getattr(row, "evidence_anchor_course", None) and not pd.isna(row.evidence_anchor_course):
        notes.append(f"Dataset sequencing also shows it commonly appears after {row.evidence_anchor_course}.")
    if getattr(row, "bundle_coreq_ready", None) == "Yes" and getattr(row, "missing_coreqs", ""):
        notes.append("This plan keeps its required corequisite pairing together.")
    return " ".join(notes)


st.title("Semester Planner")

transcript_df = st.session_state.get("transcript_df", pd.DataFrame())
if transcript_df.empty:
    st.warning("Load a transcript on the Input page to build a semester plan.")
    st.stop()

student_profile = {
    "student_id": st.session_state.get("active_student_id", "session_student"),
    "transcript_df": transcript_df,
    "interests": st.session_state.get("selected_interests", []),
    "target_credit_load": st.session_state.get("target_credit_load", 12),
}

recommendations = st.session_state.get("recommendations_df")
required_evidence_columns = {"evidence_avg_gpa", "evidence_pass_rate", "evidence_records"}
if recommendations is None or recommendations.empty or not required_evidence_columns.issubset(recommendations.columns):
    recommendations = recommend_courses(student_profile)

plan_df = build_semester_plan(recommendations, max_credits=int(student_profile["target_credit_load"]))

if plan_df.empty:
    st.info("No eligible bundle could be assembled for the current profile.")
    st.stop()

total_credits = int(plan_df["credits"].sum())
st.metric("Planned credits", total_credits)

st.dataframe(
    plan_df[
        [
            "course_number",
            "course_title",
            "credits",
            "score",
            "bundle_reason",
            "evidence_avg_gpa",
            "evidence_pass_rate",
            "bundle_top_partner",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

bundle_fig = px.pie(plan_df, names="bundle_reason", values="credits", title="Bundle composition")
st.plotly_chart(bundle_fig, use_container_width=True)

st.subheader("Bundle rationale")
for row in plan_df.itertuples():
    with st.container(border=True):
        st.markdown(f"**{row.course_number} - {row.course_title}**")
        st.write(row.explanation)
        st.caption(build_plan_evidence_line(row))
        st.caption(build_bundle_rationale(row))
