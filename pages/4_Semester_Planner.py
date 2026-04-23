from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.plot_utils import render_timeline_flowchart
from src.recommender import recommend_courses
from src.semester_planner import build_graduation_roadmap, build_semester_plan


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

st.subheader("Graduation roadmap")
st.caption(
    "Everything you've already taken (or are taking now) stays where it happened on your transcript. "
    "The teal boxes are suggested placements for the remaining degree requirements, scheduled term-by-term "
    "with prerequisites and corequisite pairs respected."
)

default_credit_load = int(st.session_state.get("target_credit_load", 12) or 12)
default_credit_load = max(3, min(18, default_credit_load))

roadmap_controls = st.columns([2, 1, 2])
with roadmap_controls[0]:
    roadmap_max_credits = st.slider(
        "Max credits per future term",
        min_value=3,
        max_value=18,
        value=default_credit_load,
        step=1,
        help="Upper bound on scheduled credit hours per future semester. Summer terms are capped at 6.",
        key="roadmap_max_credits",
    )
with roadmap_controls[1]:
    include_summer = st.toggle(
        "Include summer terms",
        value=bool(st.session_state.get("roadmap_include_summer", False)),
        help="When on, the planner can place courses in Summer sessions between Spring and Fall.",
        key="roadmap_include_summer",
    )
with roadmap_controls[2]:
    max_future_terms = st.slider(
        "Planning horizon (future terms)",
        min_value=2,
        max_value=12,
        value=int(st.session_state.get("roadmap_max_future_terms", 8) or 8),
        step=1,
        help="How many future semesters to fill before giving up and marking the rest as unscheduled.",
        key="roadmap_max_future_terms",
    )

roadmap_df = build_graduation_roadmap(
    transcript_df,
    target_credits_per_term=int(roadmap_max_credits),
    include_summer=bool(include_summer),
    max_future_terms=int(max_future_terms),
)

suggested_rows = roadmap_df[roadmap_df["status"] == "suggested"]
unscheduled_rows = roadmap_df[roadmap_df["status"] == "unscheduled"]
future_terms = sorted(suggested_rows["term_label"].unique().tolist())

roadmap_metric_cols = st.columns(4)
roadmap_metric_cols[0].metric("Courses suggested", int(len(suggested_rows)))
roadmap_metric_cols[1].metric("Future terms planned", len(future_terms))
roadmap_metric_cols[2].metric(
    "Suggested credit hours",
    f"{pd.to_numeric(suggested_rows.get('credit_hours'), errors='coerce').fillna(0).sum():.0f}",
)
roadmap_metric_cols[3].metric("Unscheduled", int(len(unscheduled_rows)))

if suggested_rows.empty and unscheduled_rows.empty:
    st.success(
        "Every requirement in the degree plan is already on your transcript (completed or in progress). "
        "Nothing left to schedule - congrats!"
    )
else:
    roadmap_fig = render_timeline_flowchart(
        roadmap_df,
        title="Graduation roadmap",
        legend_title="Course status",
        show_term_credits=True,
    )
    st.plotly_chart(roadmap_fig, use_container_width=False)
    st.caption(
        "Past and in-progress columns come straight from your transcript. Teal 'Suggested' boxes are this "
        "semester's auto-plan for remaining requirements; the term totals in each header show the planned "
        "credit load. Hover any box for details, including which requirement it satisfies. Scroll "
        "horizontally if the chart extends past the page."
    )

    if not unscheduled_rows.empty:
        st.warning(
            f"{len(unscheduled_rows)} requirement(s) could not be placed within the current horizon. "
            "Try raising the max credits per term, enabling summer, or extending the planning horizon."
        )
        st.dataframe(
            unscheduled_rows[
                [
                    "requirement_id",
                    "requirement_name",
                    "course_number",
                    "category",
                    "credit_hours",
                ]
            ].rename(columns={"credit_hours": "credits"}),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Suggested schedule table", expanded=False):
        st.dataframe(
            suggested_rows[
                [
                    "term_label",
                    "course_number",
                    "course_title",
                    "credit_hours",
                    "category",
                    "requirement_id",
                    "requirement_name",
                    "is_elective_slot",
                ]
            ].rename(columns={"credit_hours": "credits"}),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.subheader("Next-term bundle recommendations")

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
