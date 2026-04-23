from __future__ import annotations

import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit import build_transcript_progression
from src.recommender import recommend_courses
from src.semester_planner import build_remaining_semester_plan, build_semester_plan


TRANSCRIPT_STATUS_COLORS = {
    "completed": "#2F855A",
    "in_progress": "#D69E2E",
    "credit_by_exam": "#2B6CB0",
    "transfer": "#805AD5",
    "Recommended next semester": "#E53E3E",
    "Recommended future semester": "#319795",
}

TRANSCRIPT_STATUS_LABELS = {
    "completed": "Completed (in residence)",
    "in_progress": "In Progress",
    "credit_by_exam": "Credit by exam",
    "transfer": "Transfer credit",
    "Recommended next semester": "Recommended next semester",
    "Recommended future semester": "Recommended future semester",
}

CATEGORY_ORDER = {
    "Math": 1,
    "Science": 2,
    "Computing": 3,
    "General Education": 4,
    "ME Core": 5,
    "ME Lab": 6,
    "Gateway Elective": 7,
    "Other": 8,
}


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


def _compact_title(title: str, width: int = 16, max_lines: int = 2) -> str:
    words = str(title).replace("Introduction to ", "").replace("Engineering ", "Engr ").replace("and", "&")
    lines = textwrap.wrap(words, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "<br>".join(lines)


def _add_legend_swatches(fig: go.Figure, statuses: list[str]) -> None:
    for status in statuses:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 16,
                    "color": TRANSCRIPT_STATUS_COLORS.get(status, "#CBD5E0"),
                    "line": {"color": "#2D3748", "width": 1.2},
                    "symbol": "square",
                },
                name=TRANSCRIPT_STATUS_LABELS.get(status, status),
                showlegend=True,
                hoverinfo="skip",
            )
        )


def _actual_transcript_flow_rows(transcript_df: pd.DataFrame) -> pd.DataFrame:
    transcript_progression = build_transcript_progression(transcript_df)
    if transcript_progression.empty:
        return pd.DataFrame()

    ordered_terms = (
        transcript_progression[["term_order", "term_label"]]
        .drop_duplicates()
        .sort_values("term_order")
        .reset_index(drop=True)
    )
    term_to_x = {row.term_label: idx + 1 for idx, row in ordered_terms.iterrows()}
    rows = transcript_progression.copy()
    rows["x_slot"] = rows["term_label"].map(term_to_x)
    rows["x_label"] = rows["term_label"]
    rows["planned_term_order"] = rows["term_order"]
    rows["planned_semester"] = pd.NA
    rows["planner_status"] = rows["status"]
    rows["display_term"] = rows["term_label"]
    rows["score"] = pd.NA
    rows["predicted_gpa"] = pd.NA
    return rows[
        [
            "x_slot",
            "x_label",
            "planned_term_order",
            "planned_semester",
            "course_number",
            "course_title",
            "status",
            "planner_status",
            "credit_hours",
            "grade",
            "category",
            "display_term",
            "score",
            "predicted_gpa",
        ]
    ]


def _planned_flow_rows(four_year_plan_df: pd.DataFrame) -> pd.DataFrame:
    if four_year_plan_df.empty:
        return pd.DataFrame()

    rows = four_year_plan_df.copy()
    rows["status"] = rows["planner_status"]
    rows["credit_hours"] = rows["credits"]
    rows["grade"] = ""
    rows["x_label"] = rows["planned_term_label"]
    rows["display_term"] = rows["planned_term_label"]
    if "category" not in rows.columns:
        rows["category"] = "Other"
    return rows[
        [
            "x_label",
            "planned_term_order",
            "planned_semester",
            "course_number",
            "course_title",
            "status",
            "planner_status",
            "credit_hours",
            "grade",
            "category",
            "display_term",
            "score",
            "predicted_gpa",
        ]
    ]


def _build_four_year_flowchart(transcript_df: pd.DataFrame, four_year_plan_df: pd.DataFrame) -> go.Figure:
    actual_rows = _actual_transcript_flow_rows(transcript_df)
    planned_rows = _planned_flow_rows(four_year_plan_df)
    if not planned_rows.empty:
        existing_labels = actual_rows[["x_label", "x_slot"]].drop_duplicates() if not actual_rows.empty else pd.DataFrame(columns=["x_label", "x_slot"])
        next_slot = int(existing_labels["x_slot"].max()) + 1 if not existing_labels.empty else 1
        planned_labels = (
            planned_rows[["x_label", "planned_term_order"]]
            .drop_duplicates()
            .sort_values("planned_term_order")
            .reset_index(drop=True)
        )
        planned_slot_lookup = {row.x_label: next_slot + idx for idx, row in planned_labels.iterrows()}
        planned_rows["x_slot"] = planned_rows["x_label"].map(planned_slot_lookup)

    combined = pd.concat([actual_rows, planned_rows], ignore_index=True)

    if combined.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Four-year graduation progression",
            height=320,
            annotations=[
                {
                    "text": "No transcript or planner courses available to chart.",
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14, "color": "#4A5568"},
                }
            ],
        )
        return fig

    combined["_category_rank"] = combined["category"].map(CATEGORY_ORDER).fillna(99)
    combined["_is_planned"] = combined["planner_status"].astype(str).str.startswith("Recommended")
    ordered_slots = (
        combined[["x_slot", "x_label", "planned_term_order"]]
        .drop_duplicates()
        .sort_values(["x_slot", "planned_term_order"])
        .reset_index(drop=True)
    )
    max_rows = max(int((combined["x_slot"] == slot).sum()) for slot in ordered_slots["x_slot"])
    max_rows = max(max_rows, 1)

    fig = go.Figure()
    box_half_width = 0.46
    box_half_height = 0.50
    hover_rows: list[dict] = []

    for slot_row in ordered_slots.itertuples():
        x_slot = float(slot_row.x_slot)
        fig.add_shape(
            type="rect",
            x0=x_slot - 0.5,
            x1=x_slot + 0.5,
            y0=-0.7,
            y1=max_rows - 0.2,
            fillcolor="#F7FAFC",
            line={"color": "#E2E8F0", "width": 1},
            layer="below",
        )
        fig.add_annotation(
            x=x_slot,
            y=max_rows + 0.2,
            text=f"<b>{slot_row.x_label}</b>",
            showarrow=False,
            font={"size": 14, "color": "#1A202C"},
        )

        frame = (
            combined[combined["x_slot"] == slot_row.x_slot]
            .sort_values(["_is_planned", "_category_rank", "course_number"])
            .reset_index(drop=True)
        )
        for row_index, row in enumerate(frame.itertuples(), start=0):
            x_pos = float(slot_row.x_slot)
            y_pos = float(max_rows - row_index - 1)
            status = str(row.planner_status)
            color = TRANSCRIPT_STATUS_COLORS.get(status, "#CBD5E0")
            font_color = "white" if status != "Recommended future semester" else "#102A43"

            fig.add_shape(
                type="rect",
                x0=x_pos - box_half_width,
                x1=x_pos + box_half_width,
                y0=y_pos - box_half_height,
                y1=y_pos + box_half_height,
                fillcolor=color,
                line={"color": "#2D3748", "width": 1.4},
                layer="below",
            )
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=(
                    f"<b>{row.course_number}</b>"
                    f"<br><span style='font-size:9px'>{_compact_title(row.course_title)}</span>"
                ),
                showarrow=False,
                align="center",
                font={"size": 11, "color": font_color},
            )

            credits = f"{float(row.credit_hours):g} hrs" if pd.notna(row.credit_hours) else "Credits: n/a"
            grade = f"Grade: {row.grade}" if str(row.grade or "").strip() else "Grade: planned"
            score = f"Fit score: {float(row.score):.3f}" if pd.notna(row.score) else ""
            predicted = f"Predicted GPA: {float(row.predicted_gpa):.2f}" if pd.notna(row.predicted_gpa) else ""
            hover_lines = [
                f"<b>{row.course_number}</b>",
                str(row.course_title),
                f"Term: {row.display_term}",
                f"Status: {TRANSCRIPT_STATUS_LABELS.get(status, status)}",
                f"Degree semester: {int(row.planned_semester)}" if pd.notna(getattr(row, "planned_semester", pd.NA)) else "",
                credits,
                grade,
                score,
                predicted,
            ]
            hover_rows.append(
                {
                    "x": x_pos,
                    "y": y_pos,
                    "status": status,
                    "hover": "<br>".join(line for line in hover_lines if line),
                }
            )

    hover_df = pd.DataFrame(hover_rows)
    for status, frame in hover_df.groupby("status"):
        fig.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["y"],
                mode="markers",
                marker={"size": 42, "opacity": 0, "color": TRANSCRIPT_STATUS_COLORS.get(status, "#CBD5E0")},
                customdata=frame["hover"],
                hovertemplate="%{customdata}<extra></extra>",
                showlegend=False,
            )
        )

    statuses = [status for status in TRANSCRIPT_STATUS_COLORS if status in set(combined["planner_status"])]
    _add_legend_swatches(fig, statuses)

    fig.update_layout(
        height=max(720, 150 * max_rows),
        width=max(1800, 220 * len(ordered_slots)),
        title="Four-year graduation progression",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 30, "r": 220, "t": 80, "b": 30},
        xaxis={"visible": False, "range": [0.4, float(ordered_slots["x_slot"].max()) + 0.6]},
        yaxis={"visible": False, "range": [-0.9, max_rows + 0.8]},
        legend={
            "title": {"text": "<b>Course status</b>", "font": {"color": "#000000", "size": 13}},
            "bgcolor": "rgba(255,255,255,0.9)",
            "bordercolor": "#CBD5E0",
            "borderwidth": 1,
            "font": {"size": 12, "color": "#1A202C"},
            "itemsizing": "constant",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.01,
        },
        font={"size": 12},
    )
    return fig


def _planner_summary_table(plan_df: pd.DataFrame) -> pd.DataFrame:
    if plan_df.empty:
        return plan_df
    columns = [
        "planned_term_label",
        "planned_semester",
        "course_number",
        "course_title",
        "credits",
        "score",
        "predicted_gpa",
        "eligibility_status",
        "missing_coreqs",
        "bundle_reason",
    ]
    sort_columns = [column for column in ["planned_term_order", "course_number"] if column in plan_df.columns]
    available = [column for column in columns if column in plan_df.columns]
    result = plan_df.sort_values(sort_columns).reset_index(drop=True) if sort_columns else plan_df.reset_index(drop=True)
    return result[available]


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
    "student_gpa": st.session_state.get("student_gpa"),
}

recommendations = st.session_state.get("recommendations_df")
required_evidence_columns = {"evidence_avg_gpa", "evidence_pass_rate", "evidence_records"}
if recommendations is None or recommendations.empty or not required_evidence_columns.issubset(recommendations.columns):
    recommendations = recommend_courses(student_profile)

plan_df = build_semester_plan(recommendations, max_credits=int(student_profile["target_credit_load"]))
four_year_plan_df = build_remaining_semester_plan(
    student_profile,
    max_credits=int(student_profile["target_credit_load"]),
    final_semester=8,
)

next_semester_plan = pd.DataFrame()
remaining_semester_plan = pd.DataFrame()
if not four_year_plan_df.empty:
    next_semester_plan = four_year_plan_df[
        four_year_plan_df["planner_status"] == "Recommended next semester"
    ].copy()
    remaining_semester_plan = four_year_plan_df[
        four_year_plan_df["planner_status"] == "Recommended future semester"
    ].copy()

st.subheader("Four-year graduation progression")
progression_fig = _build_four_year_flowchart(transcript_df, four_year_plan_df)
st.plotly_chart(progression_fig, use_container_width=False)
st.caption(
    "Actual transcript courses keep the same colors as the Degree Audit transcript view. "
    "Recommended next-semester and later-semester courses are added as separate legend colors. "
    "Scroll horizontally if the chart extends past the page."
)

if four_year_plan_df.empty:
    st.info("No additional remaining-semester recommendations were generated for this profile.")

st.subheader("Next term recommendations")
if next_semester_plan.empty:
    st.info("No next-term bundle could be assembled for the current profile.")
else:
    st.dataframe(_planner_summary_table(next_semester_plan), use_container_width=True, hide_index=True)

st.subheader("Remaining term recommendations")
if remaining_semester_plan.empty:
    st.info("The current next-term bundle completes the generated plan.")
else:
    st.dataframe(_planner_summary_table(remaining_semester_plan), use_container_width=True, hide_index=True)

if not plan_df.empty:
    bundle_fig = px.pie(plan_df, names="bundle_reason", values="credits", title="Bundle composition")
    st.plotly_chart(bundle_fig, use_container_width=True)

    st.subheader("Bundle rationale")
    for row in plan_df.itertuples():
        with st.container(border=True):
            st.markdown(f"**{row.course_number} - {row.course_title}**")
            st.write(row.explanation)
            st.caption(build_plan_evidence_line(row))
            st.caption(build_bundle_rationale(row))
