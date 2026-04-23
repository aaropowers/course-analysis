from __future__ import annotations

import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit import (
    audit_degree_progress,
    build_degree_plan_progression,
    evaluate_course_dependencies,
    get_locked_courses,
    get_missing_requirements,
)
from src.utils import transcript_completed_hours, transcript_gpa


def _compact_title(title: str, width: int = 18, max_lines: int = 2) -> str:
    words = str(title).replace("Introduction to ", "").replace("Engineering ", "Engr ").replace("and", "&")
    lines = textwrap.wrap(words, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "<br>".join(lines)


def _build_flowchart_figure(progression_df: pd.DataFrame, dependency_edges: pd.DataFrame) -> go.Figure:
    status_colors = {
        "Completed": "#2F855A",
        "Eligible": "#3182CE",
        "Eligible with corequisite": "#DD6B20",
        "Locked": "#A0AEC0",
    }

    semester_values = sorted(progression_df["recommended_semester"].unique().tolist())
    progression_df = progression_df.copy()
    progression_df["sort_bucket"] = progression_df["category"].map(
        {
            "Math": 1,
            "Science": 2,
            "Computing": 3,
            "General Education": 4,
            "ME Core": 5,
            "ME Lab": 6,
            "Gateway Elective": 7,
        }
    ).fillna(99)

    node_positions: dict[str, tuple[float, float]] = {}
    course_to_node_key: dict[str, str] = {}
    hover_rows = []
    max_rows = 0

    for semester in semester_values:
        frame = progression_df[progression_df["recommended_semester"] == semester].sort_values(
            ["sort_bucket", "requirement_id"]
        )
        max_rows = max(max_rows, len(frame))
        for row_index, row in enumerate(frame.itertuples(), start=1):
            x_pos = float(semester)
            y_pos = float(max_rows - row_index)
            node_positions[row.requirement_id] = (x_pos, y_pos)
            course_to_node_key.setdefault(row.course_number, row.requirement_id)
            hover_lines = [
                f"<b>{row.requirement_id}</b>",
                row.requirement_name,
                f"Course: {row.course_number}",
                f"Status: {row.status}",
            ]
            if row.matched_course:
                hover_lines.append(f"Transcript match: {row.matched_course}")
            if row.missing_prereqs:
                hover_lines.append(f"Missing prereqs: {row.missing_prereqs}")
            if row.missing_coreqs:
                hover_lines.append(f"Missing coreqs: {row.missing_coreqs}")
            hover_rows.append(
                {
                    "requirement_id": row.requirement_id,
                    "course_number": row.course_number,
                    "status": row.status,
                    "x": x_pos,
                    "y": y_pos,
                    "hover": "<br>".join(hover_lines),
                }
            )

    # Recompute y positions with a shared top-to-bottom row count so columns align cleanly.
    node_positions.clear()
    hover_rows = []
    for semester in semester_values:
        frame = progression_df[progression_df["recommended_semester"] == semester].sort_values(
            ["sort_bucket", "requirement_id"]
        ).reset_index(drop=True)
        for row_index, row in enumerate(frame.itertuples(), start=0):
            x_pos = float(semester)
            y_pos = float(max_rows - row_index - 1)
            node_positions[row.requirement_id] = (x_pos, y_pos)
            course_to_node_key.setdefault(row.course_number, row.requirement_id)
            hover_lines = [
                f"<b>{row.requirement_id}</b>",
                row.requirement_name,
                f"Course: {row.course_number}",
                f"Status: {row.status}",
            ]
            if row.matched_course:
                hover_lines.append(f"Transcript match: {row.matched_course}")
            if row.missing_prereqs:
                hover_lines.append(f"Missing prereqs: {row.missing_prereqs}")
            if row.missing_coreqs:
                hover_lines.append(f"Missing coreqs: {row.missing_coreqs}")
            hover_rows.append(
                {
                    "requirement_id": row.requirement_id,
                    "course_number": row.course_number,
                    "status": row.status,
                    "x": x_pos,
                    "y": y_pos,
                    "hover": "<br>".join(hover_lines),
                }
            )

    fig = go.Figure()
    box_half_width = 0.42
    box_half_height = 0.36

    # Semester column backgrounds
    for semester in semester_values:
        fig.add_shape(
            type="rect",
            x0=semester - 0.48,
            x1=semester + 0.48,
            y0=-0.6,
            y1=max_rows - 0.2,
            fillcolor="#F7FAFC",
            line={"color": "#E2E8F0", "width": 1},
            layer="below",
        )
        fig.add_annotation(
            x=semester,
            y=max_rows + 0.15,
            text=f"<b>Semester {semester}</b>",
            showarrow=False,
            font={"size": 13, "color": "#1A202C"},
        )

    for row in progression_df.itertuples():
        x_pos, y_pos = node_positions[row.requirement_id]
        fill_color = status_colors.get(row.status, "#CBD5E0")
        fig.add_shape(
            type="rect",
            x0=x_pos - box_half_width,
            x1=x_pos + box_half_width,
            y0=y_pos - box_half_height,
            y1=y_pos + box_half_height,
            fillcolor=fill_color,
            line={"color": "#2D3748", "width": 1.4},
            layer="below",
        )
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=f"<b>{row.course_number}</b><br><span style='font-size:10px'>{_compact_title(row.requirement_name)}</span>",
            showarrow=False,
            align="center",
            font={"size": 11, "color": "white" if row.status != "Locked" else "#1A202C"},
        )

    for row in dependency_edges.itertuples():
        source_key = course_to_node_key.get(row.prerequisite_course)
        target_key = course_to_node_key.get(row.course_number)
        if not source_key or not target_key:
            continue
        x0, y0 = node_positions[source_key]
        x1, y1 = node_positions[target_key]
        fig.add_annotation(
            x=x1 - box_half_width,
            y=y1,
            ax=x0 + box_half_width,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5 if row.prereq_type == "PREREQ" else 1.2,
            arrowcolor="#718096" if row.prereq_type == "PREREQ" else "#DD6B20",
            opacity=0.85,
            standoff=2,
            startstandoff=2,
        )

    hover_df = pd.DataFrame(hover_rows)
    for status, frame in hover_df.groupby("status"):
        fig.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["y"],
                mode="markers",
                marker={"size": 42, "opacity": 0},
                customdata=frame["hover"],
                hovertemplate="%{customdata}<extra></extra>",
                name=status,
                showlegend=True,
            )
        )

    fig.update_layout(
        height=max(720, 130 * max_rows),
        title="Mechanical engineering degree flow",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        xaxis={
            "visible": False,
            "range": [min(semester_values) - 0.6, max(semester_values) + 0.6],
        },
        yaxis={
            "visible": False,
            "range": [-0.8, max_rows + 0.6],
        },
        legend_title="Box status",
    )
    return fig


st.title("Degree Audit")

transcript_df = st.session_state.get("transcript_df", pd.DataFrame())
if transcript_df.empty:
    st.warning("Load a transcript on the Input page to see the degree audit.")
    st.stop()

audit_result = audit_degree_progress(transcript_df)
missing_df = get_missing_requirements(audit_result)
locked_df = get_locked_courses(transcript_df)
progression_df, dependency_edges = build_degree_plan_progression(transcript_df)
dependency_debug_df = evaluate_course_dependencies(transcript_df)

metric_cols = st.columns(5)
metric_cols[0].metric("Progress", f"{audit_result['progress_percent']}%")
metric_cols[1].metric(
    "Degree Plan Hours",
    f"{audit_result['completed_hours']} / {audit_result['total_hours']}",
)
metric_cols[2].metric("UT total hours taken", transcript_completed_hours(transcript_df))
metric_cols[3].metric("Remaining requirements", len(missing_df))
gpa = transcript_gpa(transcript_df)
metric_cols[4].metric("Transcript GPA", gpa if gpa is not None else "N/A")
st.caption(
    "UT total hours taken matches the Academic Summary total: "
    "completed in-residence plus credit-by-exam credits with hours greater than zero. "
    "Transfer and in-progress lines are excluded."
)

summary_fig = px.bar(
    audit_result["summary"],
    x="category",
    y=[column for column in audit_result["summary"].columns if column != "category"],
    title="Requirement status by category",
    barmode="group",
)
st.plotly_chart(summary_fig, use_container_width=True)

st.subheader("Degree progression map")
progress_fig = _build_flowchart_figure(progression_df, dependency_edges)
st.plotly_chart(progress_fig, use_container_width=True)
st.caption(
    "Grey arrows are prerequisites. Orange arrows mark same-term corequisite relationships. "
    "Hover any box to inspect the exact missing dependency state."
)

st.subheader("Requirement checklist")
st.dataframe(audit_result["requirements"], use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Remaining requirements")
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Locked courses")
    st.dataframe(
        locked_df[["course_number", "course_title", "missing_prereqs", "missing_coreqs"]].head(12),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Dependency schema debugger", expanded=False):
    st.caption(
        "Use this table to verify whether a course is blocked by hard prerequisites or only waiting on a corequisite."
    )
    st.dataframe(
        dependency_debug_df[
            [
                "course_number",
                "course_title",
                "recommended_semester",
                "eligibility_status",
                "missing_prereqs",
                "missing_coreqs",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
