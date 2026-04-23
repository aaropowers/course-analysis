from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.audit import (
    audit_degree_progress,
    build_degree_plan_progression,
    build_transcript_progression,
    evaluate_course_dependencies,
    get_locked_courses,
    get_missing_requirements,
)
from src.plot_utils import (
    TIMELINE_STATUS_COLORS,
    TIMELINE_STATUS_LABELS,
    add_legend_swatches,
    compact_title,
    render_timeline_flowchart,
)
from src.utils import (
    build_course_lookup,
    split_allowed_courses,
    transcript_completed_hours,
    transcript_gpa,
)


REQUIREMENT_STATUS_COLORS = {
    "Completed": "#2F855A",
    "In Progress": "#D69E2E",
    "Eligible": "#3182CE",
    "Eligible with corequisite": "#DD6B20",
    "Locked": "#A0AEC0",
}

# Back-compat aliases for any external readers / notebooks that imported these.
TRANSCRIPT_STATUS_COLORS = TIMELINE_STATUS_COLORS
TRANSCRIPT_STATUS_LABELS = TIMELINE_STATUS_LABELS


def _format_elective_options(allowed_courses_raw: str, catalog_lookup: dict) -> list[str]:
    options = sorted(split_allowed_courses(allowed_courses_raw))
    lines: list[str] = []
    for code in options:
        title = catalog_lookup.get(code, {}).get("course_title", "")
        lines.append(f"  • {code} - {title}" if title else f"  • {code}")
    return lines


def _build_flowchart_figure(progression_df: pd.DataFrame) -> go.Figure:
    status_colors = REQUIREMENT_STATUS_COLORS
    catalog_lookup = build_course_lookup()

    def is_elective_slot(row) -> bool:
        return str(getattr(row, "category", "")).strip() == "Gateway Elective"

    def display_heading(row) -> str:
        if is_elective_slot(row):
            return "Career Gateway Elective"
        return str(row.course_number)

    def display_subtitle(row) -> str:
        if is_elective_slot(row):
            slot_label = str(row.requirement_name or "").replace("Gateway Elective", "Slot").strip() or "Slot"
            if row.matched_course:
                return f"{slot_label} - {row.matched_course}"
            return slot_label
        return compact_title(row.requirement_name)

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
                f"Status: {row.status}",
            ]
            if is_elective_slot(row):
                option_lines = _format_elective_options(row.allowed_courses, catalog_lookup)
                if option_lines:
                    hover_lines.append("Options:")
                    hover_lines.extend(option_lines)
                if row.matched_course:
                    hover_lines.append(f"Transcript match: {row.matched_course}")
            else:
                hover_lines.append(f"Course: {row.course_number}")
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
    box_half_width = 0.46
    box_half_height = 0.44

    # Semester column backgrounds
    for semester in semester_values:
        fig.add_shape(
            type="rect",
            x0=semester - 0.5,
            x1=semester + 0.5,
            y0=-0.7,
            y1=max_rows - 0.2,
            fillcolor="#F7FAFC",
            line={"color": "#E2E8F0", "width": 1},
            layer="below",
        )
        fig.add_annotation(
            x=semester,
            y=max_rows + 0.2,
            text=f"<b>Semester {semester}</b>",
            showarrow=False,
            font={"size": 14, "color": "#1A202C"},
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
        heading = display_heading(row)
        subtitle = display_subtitle(row)
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=f"<b>{heading}</b><br><span style='font-size:11px'>{subtitle}</span>",
            showarrow=False,
            align="center",
            font={"size": 12, "color": "white" if row.status != "Locked" else "#1A202C"},
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
                showlegend=False,
            )
        )

    statuses_present = {
        status: REQUIREMENT_STATUS_COLORS.get(status, "#CBD5E0")
        for status in progression_df["status"].unique().tolist()
    }
    add_legend_swatches(fig, statuses_present)

    width = max(1700, 220 * max(len(semester_values), 1))
    fig.update_layout(
        height=max(900, 170 * max_rows),
        width=width,
        title="Mechanical engineering degree flow",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 30, "r": 180, "t": 80, "b": 30},
        xaxis={
            "visible": False,
            "range": [min(semester_values) - 0.6, max(semester_values) + 0.6],
        },
        yaxis={
            "visible": False,
            "range": [-0.9, max_rows + 0.8],
        },
        legend={
            "title": {
                "text": "<b>Requirement status</b>",
                "font": {"color": "#000000", "size": 13},
            },
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


st.title("Degree Audit")

transcript_df = st.session_state.get("transcript_df", pd.DataFrame())
if transcript_df.empty:
    st.warning("Load a transcript on the Input page to see the degree audit.")
    st.stop()

audit_result = audit_degree_progress(transcript_df)
missing_df = get_missing_requirements(audit_result)
locked_df = get_locked_courses(transcript_df)
progression_df, dependency_edges = build_degree_plan_progression(transcript_df)
transcript_progression_df = build_transcript_progression(transcript_df)
dependency_debug_df = evaluate_course_dependencies(transcript_df)

combined_progress_percent = round(
    audit_result["progress_percent"] + audit_result.get("in_progress_percent", 0.0),
    1,
)
combined_hours = audit_result["completed_hours"] + audit_result.get("in_progress_hours", 0)

metric_cols = st.columns(5)
metric_cols[0].metric("Progress", f"{combined_progress_percent}%")
metric_cols[1].metric(
    "Degree Plan Hours",
    f"{combined_hours} / {audit_result['total_hours']}",
)
metric_cols[2].metric("UT In Residence hours taken", transcript_completed_hours(transcript_df))
metric_cols[3].metric("Remaining requirements", len(missing_df))
gpa = transcript_gpa(transcript_df)
metric_cols[4].metric("Transcript GPA", gpa if gpa is not None else "N/A")
st.caption(
    "NOTE:"
    "Transfer and in-progress lines are excluded from In Residence Hours"
)

status_column_order = ["Completed", "In Progress", "Eligible", "Eligible with corequisite", "Locked", "Remaining"]
available_status_columns = [
    column
    for column in status_column_order
    if column in audit_result["summary"].columns
] + [
    column
    for column in audit_result["summary"].columns
    if column not in status_column_order and column != "category"
]

summary_fig = px.bar(
    audit_result["summary"],
    x="category",
    y=available_status_columns,
    title="Requirement status by category",
    barmode="stack",
    color_discrete_map=REQUIREMENT_STATUS_COLORS,
)
summary_fig.update_layout(
    xaxis={"type": "category", "tickangle": 0, "title": ""},
    yaxis={"title": "Requirements", "dtick": 1},
    legend_title="Status",
    bargap=0.25,
    margin={"l": 30, "r": 30, "t": 70, "b": 30},
)
st.plotly_chart(summary_fig, use_container_width=True)

st.subheader("Degree progression map")

taken_count = int(len(transcript_progression_df[transcript_progression_df["status"] != "in_progress"]))
in_progress_count = int(len(transcript_progression_df[transcript_progression_df["status"] == "in_progress"]))
remaining_count = int(len(missing_df))

map_summary_cols = st.columns(3)
map_summary_cols[0].metric("Courses taken", taken_count)
map_summary_cols[1].metric("Courses in progress", in_progress_count)
map_summary_cols[2].metric("Requirements remaining", remaining_count)

map_view = st.radio(
    "Map view",
    options=("Degree Requirements View", "Transcript View"),
    index=0,
    horizontal=True,
    help=(
        "Degree Requirements View shows every requirement in the degree plan with its current status "
        "based on your transcript. Transcript View groups the actual courses you took by the term you "
        "took them in (e.g., Fall 2022, Spring 2023)."
    ),
    key="degree_audit_map_view",
)

if map_view == "Degree Requirements View":
    progress_fig = _build_flowchart_figure(progression_df)
    st.plotly_chart(progress_fig, use_container_width=False)
    st.caption(
        "Each box is a requirement colored by status (Completed, In Progress, Eligible, "
        "Eligible with corequisite, or Locked). Hover any box to see the matched transcript course "
        "and any missing prerequisites or corequisites. Career Gateway Elective slots show the full "
        "list of allowed courses on hover. Scroll the chart horizontally if it extends past the page."
    )
else:
    transcript_fig = render_timeline_flowchart(
        transcript_progression_df,
        title="Transcript timeline",
        legend_title="Course status",
        empty_message="No transcript courses available to chart.",
    )
    st.plotly_chart(transcript_fig, use_container_width=False)
    st.caption(
        "Each column is a term from your transcript. Boxes are colored by source: in-residence, "
        "credit by exam, transfer, or in progress. Hover any box to see the grade and credit hours. "
        "Scroll the chart horizontally if it extends past the page."
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
