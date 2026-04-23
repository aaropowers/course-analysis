"""Shared Plotly helpers for the term-by-term flowchart.

Both the Degree Audit Transcript View and the Semester Planner Graduation
Roadmap render the same style of chart: one column per term, one box per
course, status-based colors, and a visible legend. Putting the renderer in a
single module keeps the two pages in sync and avoids drift in styling.

Roadmap input rows may carry additional columns (``requirement_id``,
``requirement_name``, ``is_elective_slot``, ``allowed_courses``) and new
statuses (``suggested`` for future-term placements and ``unscheduled`` for
requirements the scheduler could not fit). The renderer tolerates missing
optional columns so plain transcript frames still work unchanged.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import plotly.graph_objects as go


TIMELINE_STATUS_COLORS: dict[str, str] = {
    "completed": "#2F855A",
    "in_progress": "#D69E2E",
    "credit_by_exam": "#2B6CB0",
    "transfer": "#805AD5",
    "suggested": "#319795",
    "unscheduled": "#E53E3E",
}

TIMELINE_STATUS_LABELS: dict[str, str] = {
    "completed": "Completed (in residence)",
    "in_progress": "In Progress",
    "credit_by_exam": "Credit by exam",
    "transfer": "Transfer credit",
    "suggested": "Suggested (future term)",
    "unscheduled": "Unscheduled",
}

_CATEGORY_RANK = {
    "Math": 1,
    "Science": 2,
    "Computing": 3,
    "General Education": 4,
    "ME Core": 5,
    "ME Lab": 6,
    "Gateway Elective": 7,
    "Other": 8,
}


def compact_title(title: str, width: int = 20, max_lines: int = 3) -> str:
    """Wrap a course title onto a few short lines so it fits inside a node box."""
    words = (
        str(title)
        .replace("Introduction to ", "")
        .replace("Engineering ", "Engr ")
        .replace("and", "&")
    )
    lines = textwrap.wrap(words, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "<br>".join(lines)


def add_legend_swatches(
    fig: go.Figure,
    color_map: dict[str, str],
    label_map: dict[str, str] | None = None,
) -> None:
    """Append visible legend-only markers so the status legend stays readable.

    Rectangle nodes are drawn as Plotly shapes and the hover layer uses
    transparent scatter markers, so without these dummy traces the legend
    swatches inherit the transparent style and disappear.
    """
    for status, color in color_map.items():
        display = (label_map or {}).get(status, status)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 16,
                    "color": color,
                    "line": {"color": "#2D3748", "width": 1.2},
                    "symbol": "square",
                },
                name=display,
                showlegend=True,
                hoverinfo="skip",
            )
        )


def _format_credit_hours(value: object) -> str:
    if value in (None, ""):
        return "Credits: n/a"
    try:
        if pd.isna(value):
            return "Credits: n/a"
    except TypeError:
        pass
    try:
        return f"{float(value):g} hrs"
    except (TypeError, ValueError):
        return "Credits: n/a"


def _sum_term_credits(frame: pd.DataFrame) -> float:
    credits = pd.to_numeric(frame.get("credit_hours"), errors="coerce")
    total = float(credits.fillna(0).sum()) if credits is not None else 0.0
    return round(total, 1)


def _build_hover(row, status_labels: dict[str, str]) -> str:
    """Assemble the hover tooltip text for a single course node."""
    status = getattr(row, "status", "")
    status_display = status_labels.get(status, status)

    lines: list[str] = [f"<b>{row.course_number}</b>"]
    title = getattr(row, "course_title", "") or ""
    if title:
        lines.append(title)
    lines.append(f"Term: {row.term_label}")
    lines.append(f"Status: {status_display}")

    lines.append(_format_credit_hours(getattr(row, "credit_hours", None)))

    grade = getattr(row, "grade", None)
    if status == "suggested":
        lines.append("Grade: to be earned")
    elif status == "unscheduled":
        lines.append("Grade: not yet planned")
    elif grade:
        lines.append(f"Grade: {grade}")
    else:
        lines.append("Grade: in progress")

    source = getattr(row, "source_type", "") or ""
    if source:
        lines.append(f"Source: {source}")

    requirement_name = getattr(row, "requirement_name", None)
    if requirement_name is not None and not pd.isna(requirement_name) and str(requirement_name).strip():
        lines.append(f"Satisfies: {requirement_name}")

    is_elective_raw = getattr(row, "is_elective_slot", False)
    try:
        is_elective = bool(is_elective_raw) if not pd.isna(is_elective_raw) else False
    except (TypeError, ValueError):
        is_elective = bool(is_elective_raw)
    allowed = getattr(row, "allowed_courses", None)
    if is_elective and allowed is not None and not pd.isna(allowed):
        options = [code.strip() for code in str(allowed).split("|") if code.strip()]
        if options:
            lines.append("Other allowed courses: " + ", ".join(options[:6]))
            if len(options) > 6:
                lines[-1] += f", +{len(options) - 6} more"

    return "<br>".join(line for line in lines if line)


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=320,
        annotations=[
            {
                "text": message,
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


def render_timeline_flowchart(
    timeline_df: pd.DataFrame,
    *,
    status_colors: dict[str, str] | None = None,
    status_labels: dict[str, str] | None = None,
    title: str = "Transcript timeline",
    legend_title: str = "Course status",
    empty_message: str = "No courses available to chart.",
    show_term_credits: bool = False,
) -> go.Figure:
    """Render a term-by-term flowchart from a timeline DataFrame.

    The DataFrame must have the columns emitted by
    :func:`src.audit.build_transcript_progression` (``term``, ``term_label``,
    ``term_order``, ``course_number``, ``course_title``, ``status``,
    ``credit_hours``, ``grade``, ``source_type``, ``category``). Roadmap rows
    may additionally supply ``requirement_id``, ``requirement_name``,
    ``is_elective_slot``, and ``allowed_courses`` which are surfaced in hover.
    """
    status_colors = status_colors or TIMELINE_STATUS_COLORS
    status_labels = status_labels or TIMELINE_STATUS_LABELS

    if timeline_df is None or timeline_df.empty:
        return _empty_figure(title, empty_message)

    df = timeline_df.copy()
    for column in ("requirement_id", "requirement_name", "allowed_courses"):
        if column not in df.columns:
            df[column] = pd.NA
    if "is_elective_slot" not in df.columns:
        df["is_elective_slot"] = False

    ordered_terms = (
        df[["term_order", "term_label"]]
        .drop_duplicates()
        .sort_values("term_order")
        .reset_index(drop=True)
    )
    term_to_x = {row.term_label: idx + 1 for idx, row in ordered_terms.iterrows()}
    df["x_pos"] = df["term_label"].map(term_to_x)

    max_rows = max(
        (int((df["term_label"] == label).sum()) for label in ordered_terms["term_label"]),
        default=0,
    )
    max_rows = max(max_rows, 1)

    df["_category_rank"] = df["category"].map(_CATEGORY_RANK).fillna(99)
    df = df.sort_values(["x_pos", "_category_rank", "course_number"]).reset_index(drop=True)

    fig = go.Figure()
    box_half_width = 0.46
    box_half_height = 0.44

    for label, x_pos in term_to_x.items():
        fig.add_shape(
            type="rect",
            x0=x_pos - 0.5,
            x1=x_pos + 0.5,
            y0=-0.7,
            y1=max_rows - 0.2,
            fillcolor="#F7FAFC",
            line={"color": "#E2E8F0", "width": 1},
            layer="below",
        )
        header_text = f"<b>{label}</b>"
        if show_term_credits:
            term_frame = df[df["term_label"] == label]
            total_credits = _sum_term_credits(term_frame)
            if total_credits > 0:
                header_text = (
                    f"<b>{label}</b>"
                    f"<br><span style='font-size:11px;color:#4A5568'>{total_credits:g} credits</span>"
                )
        fig.add_annotation(
            x=x_pos,
            y=max_rows + 0.35,
            text=header_text,
            showarrow=False,
            align="center",
            font={"size": 13, "color": "#1A202C"},
        )

    hover_rows: list[dict] = []
    for x_pos in sorted(df["x_pos"].unique()):
        frame = df[df["x_pos"] == x_pos].reset_index(drop=True)
        for row_index, row in enumerate(frame.itertuples(), start=0):
            y_pos = float(max_rows - row_index - 1)
            fill_color = status_colors.get(row.status, "#CBD5E0")
            fig.add_shape(
                type="rect",
                x0=float(x_pos) - box_half_width,
                x1=float(x_pos) + box_half_width,
                y0=y_pos - box_half_height,
                y1=y_pos + box_half_height,
                fillcolor=fill_color,
                line={"color": "#2D3748", "width": 1.4},
                layer="below",
            )
            subtitle = compact_title(getattr(row, "course_title", "") or "", width=18, max_lines=3)
            fig.add_annotation(
                x=float(x_pos),
                y=y_pos,
                text=(
                    f"<b>{row.course_number}</b>"
                    f"<br><span style='font-size:10px'>{subtitle}</span>"
                ),
                showarrow=False,
                align="center",
                font={"size": 12, "color": "white"},
            )
            hover_rows.append(
                {
                    "course_number": row.course_number,
                    "status": row.status,
                    "x": float(x_pos),
                    "y": y_pos,
                    "hover": _build_hover(row, status_labels),
                }
            )

    hover_df = pd.DataFrame(hover_rows)
    if not hover_df.empty:
        for status, frame in hover_df.groupby("status"):
            fig.add_trace(
                go.Scatter(
                    x=frame["x"],
                    y=frame["y"],
                    mode="markers",
                    marker={
                        "size": 42,
                        "opacity": 0,
                        "color": status_colors.get(status, "#CBD5E0"),
                    },
                    customdata=frame["hover"],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                )
            )

    statuses_present = {
        status: status_colors.get(status, "#CBD5E0")
        for status in df["status"].unique().tolist()
    }
    add_legend_swatches(fig, statuses_present, status_labels)

    term_x_values = list(term_to_x.values())
    per_column_px = 260 if show_term_credits else 240
    width = max(1700, per_column_px * max(len(term_x_values), 1))
    fig.update_layout(
        height=max(900, 170 * max_rows) + 30,
        width=width,
        title={"text": title, "y": 0.985, "yanchor": "top"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 30, "r": 220, "t": 120, "b": 30},
        xaxis={
            "visible": False,
            "range": [min(term_x_values) - 0.6, max(term_x_values) + 0.6],
        },
        yaxis={
            "visible": False,
            "range": [-0.9, max_rows + 1.2],
        },
        legend={
            "title": {
                "text": f"<b>{legend_title}</b>",
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
