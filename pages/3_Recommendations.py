from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.audit import get_eligible_courses
from src.db import log_recommendations
from src.recommender import recommend_courses
from src.utils import dataframe_fingerprint


CONFIDENCE_COLORS = {
    "High": "#2F855A",
    "Medium": "#D69E2E",
    "Low": "#C05621",
}


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


def format_prediction_range(row) -> str:
    if pd.isna(row.predicted_gpa_low) or pd.isna(row.predicted_gpa_high):
        return "N/A"
    return f"{row.predicted_gpa_low:.2f}-{row.predicted_gpa_high:.2f}"


def build_prediction_line(row) -> str:
    if pd.isna(row.predicted_gpa) or pd.isna(row.predicted_gpa_low) or pd.isna(row.predicted_gpa_high):
        return "Personalized GPA prediction is unavailable for this course."
    return (
        f"Your GPA profile suggests a likely range of {row.predicted_gpa_low:.2f}-{row.predicted_gpa_high:.2f}, "
        f"centered around {row.predicted_gpa:.2f}. Confidence: {row.prediction_confidence}. "
        f"Basis: {row.prediction_basis}."
    )


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def top_signal_summary(row) -> str:
    signals = [
        ("degree requirement", row.requirement_priority),
        ("timing fit", row.semester_fit),
        ("similar completed courses", row.similarity),
        ("common course pairing", row.coenrollment),
        ("similar students", row.collaborative),
        ("interest match", row.interest_match),
    ]
    ranked = [label for label, value in sorted(signals, key=lambda item: item[1], reverse=True) if value > 0]
    if not ranked:
        return "Unlocked course that keeps your plan moving."
    return "Top signals: " + ", ".join(ranked[:3]) + "."


def recommendation_table(recommendations: pd.DataFrame, top_n: int) -> pd.DataFrame:
    table_rows = []
    for rank, row in enumerate(recommendations.head(top_n).itertuples(), start=1):
        records = int(row.evidence_records) if not pd.isna(row.evidence_records) else 0
        coreqs = clean_text(row.missing_coreqs)
        table_rows.append(
            {
                "Rank": rank,
                "Course": row.course_number,
                "Title": row.course_title,
                "Status": clean_text(row.eligibility_status) or "Eligible",
                "Credits": int(row.credits),
                "Predicted GPA": row.predicted_gpa,
                "GPA Range": format_prediction_range(row),
                "Confidence": clean_text(row.prediction_confidence) or "N/A",
                "Fit Score": row.score,
                "Evidence": f"{records} records | {format_stat(row.evidence_pass_rate, '.1f', '%')} pass",
                "Best Signals": top_signal_summary(row).replace("Top signals: ", "").replace(".", ""),
                "Coreq Note": f"Take with {coreqs}" if coreqs else "",
            }
        )
    return pd.DataFrame(table_rows)


def render_recommendation_table(recommendations: pd.DataFrame, top_n: int) -> None:
    table_df = recommendation_table(recommendations, top_n)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=min(500, 88 + 38 * len(table_df)),
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
            "Course": st.column_config.TextColumn("Course", width="small"),
            "Title": st.column_config.TextColumn("Course title", width="large"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Credits": st.column_config.NumberColumn("Credits", width="small", format="%d"),
            "Predicted GPA": st.column_config.NumberColumn("Predicted GPA", width="small", format="%.2f"),
            "GPA Range": st.column_config.TextColumn("Likely range", width="small"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Fit Score": st.column_config.ProgressColumn(
                "Fit score",
                width="medium",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Evidence": st.column_config.TextColumn("Historical evidence", width="medium"),
            "Best Signals": st.column_config.TextColumn("Why it ranks well", width="large"),
            "Coreq Note": st.column_config.TextColumn("Coreq note", width="medium"),
        },
    )


def build_prediction_interval_chart(recommendations: pd.DataFrame, top_n: int) -> go.Figure:
    chart_df = prepare_chart_data(recommendations, top_n)
    height = max(500, 90 * len(chart_df) + 130)
    min_low = pd.to_numeric(chart_df["predicted_gpa_low"], errors="coerce").min()
    max_high = pd.to_numeric(chart_df["predicted_gpa_high"], errors="coerce").max()
    x_min = 3.0 if pd.isna(min_low) or min_low >= 3.0 else max(0.0, float(min_low) - 0.15)
    x_max = 4.05 if pd.isna(max_high) else min(4.05, max(float(max_high) + 0.08, x_min + 0.45))
    fig = go.Figure()

    grade_bands = [
        (3.7, 4.0, "A range", "rgba(47, 133, 90, 0.10)"),
        (2.7, 3.7, "B range", "rgba(214, 158, 46, 0.10)"),
        (1.7, 2.7, "C range", "rgba(192, 86, 33, 0.08)"),
    ]
    for x0, x1, label, color in grade_bands:
        visible_x0 = max(x0, x_min)
        visible_x1 = min(x1, x_max)
        if visible_x0 >= visible_x1:
            continue
        fig.add_vrect(
            x0=visible_x0,
            x1=visible_x1,
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )

    for confidence, color in CONFIDENCE_COLORS.items():
        subset = chart_df[chart_df["prediction_confidence"].eq(confidence)]
        if subset.empty:
            continue
        range_x = []
        range_y = []
        for row in subset.itertuples():
            range_x.extend([row.predicted_gpa_low, row.predicted_gpa_high, None])
            range_y.extend([row.course_label, row.course_label, None])
        fig.add_trace(
            go.Scatter(
                x=range_x,
                y=range_y,
                mode="lines",
                line={"color": color, "width": 15},
                name=f"{confidence} range",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=chart_df["predicted_gpa"],
            y=chart_df["course_label"],
            mode="markers",
            marker={
                "color": chart_df["prediction_confidence"].map(CONFIDENCE_COLORS).fillna("#4A5568"),
                "size": 18,
                "symbol": "diamond",
                "line": {"color": "white", "width": 2},
            },
            name="Predicted GPA",
            customdata=chart_df[
                [
                    "course_title",
                    "score",
                    "evidence_records",
                    "prediction_confidence",
                    "prediction_basis",
                    "predicted_gpa_low",
                    "predicted_gpa_high",
                ]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Predicted GPA: %{x:.2f}<br>"
                "Range: %{customdata[5]:.2f}-%{customdata[6]:.2f}<br>"
                "Score: %{customdata[1]:.3f}<br>"
                "Evidence records: %{customdata[2]}<br>"
                "Confidence: %{customdata[3]}<br>"
                "Basis: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    student_gpa = chart_df["student_gpa"].dropna()
    if not student_gpa.empty:
        fig.add_vline(
            x=float(student_gpa.iloc[0]),
            line_dash="dash",
            line_color="#2D3748",
            annotation_text="Your GPA",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title="Expected GPA Range for Recommended Courses",
        height=height,
        xaxis_title="Predicted GPA",
        yaxis_title="",
        xaxis={"range": [x_min, x_max], "dtick": 0.1, "tickformat": ".2f"},
        yaxis={"categoryorder": "array", "categoryarray": chart_df["course_label"].tolist()},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 85, "b": 45},
    )
    return fig


def prepare_chart_data(recommendations: pd.DataFrame, top_n: int) -> pd.DataFrame:
    chart_df = recommendations.head(top_n).copy()
    chart_df["course_label"] = chart_df["course_number"] + " - " + chart_df["course_title"]
    chart_df["range_width"] = chart_df["predicted_gpa_high"] - chart_df["predicted_gpa_low"]
    return chart_df.sort_values(["score", "predicted_gpa"], ascending=[True, True])


def build_score_driver_chart(recommendations: pd.DataFrame, top_n: int) -> go.Figure:
    chart_df = recommendations.head(top_n).copy()
    chart_df = chart_df.sort_values("score", ascending=True)
    driver_columns = [
        "requirement_priority",
        "semester_fit",
        "similarity",
        "coenrollment",
        "collaborative",
        "interest_match",
    ]
    driver_labels = {
        "requirement_priority": "Requirement",
        "semester_fit": "Timing",
        "similarity": "Similar courses",
        "coenrollment": "Common pairing",
        "collaborative": "Similar students",
        "interest_match": "Interest match",
    }
    fig = go.Figure()
    colors = ["#2F855A", "#4C78A8", "#805AD5", "#DD6B20", "#319795", "#D69E2E"]
    for column, color in zip(driver_columns, colors):
        fig.add_trace(
            go.Bar(
                x=chart_df[column],
                y=chart_df["course_number"],
                orientation="h",
                name=driver_labels[column],
                marker_color=color,
                customdata=chart_df[["course_title", "score", "predicted_gpa"]],
                hovertemplate=(
                    "<b>%{y} - %{customdata[0]}</b><br>"
                    f"{driver_labels[column]} signal: " + "%{x:.2f}<br>"
                    "Total score: %{customdata[1]:.3f}<br>"
                    "Predicted GPA: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="What is driving each recommendation?",
        barmode="stack",
        height=max(430, 58 * len(chart_df) + 120),
        xaxis_title="Signal strength",
        yaxis_title="",
        xaxis={"range": [0, max(1.0, float(chart_df[driver_columns].sum(axis=1).max()) + 0.2)]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 20, "r": 20, "t": 70, "b": 45},
    )
    return fig


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
    "student_gpa": st.session_state.get("student_gpa"),
}

eligible_df = get_eligible_courses(transcript_df)
recommendation_cache_key = (
    dataframe_fingerprint(transcript_df, columns=["course_number", "grade", "term", "year", "semester", "grade_points"]),
    tuple(sorted(student_profile["interests"])),
    int(student_profile["target_credit_load"]),
)
if st.session_state.get("recommendations_cache_key") == recommendation_cache_key:
    recommendations = st.session_state.get("recommendations_df", pd.DataFrame()).copy()
else:
    recommendations = recommend_courses(student_profile, eligible_courses=eligible_df)
    st.session_state["recommendations_df"] = recommendations
    st.session_state["recommendations_cache_key"] = recommendation_cache_key

log_recommendations(student_profile["student_id"], recommendations.head(5))

top_n = min(8, len(recommendations))
if top_n == 0:
    st.info("No eligible recommendations are available yet.")
    st.stop()

display_recommendations = recommendations.copy()
display_recommendations["predicted_gpa_range"] = display_recommendations.apply(format_prediction_range, axis=1)
display_recommendations["prediction_range_width"] = (
    display_recommendations["predicted_gpa_high"] - display_recommendations["predicted_gpa_low"]
)

st.subheader("Top next-course recommendations")
render_recommendation_table(display_recommendations, top_n)

st.subheader("Predicted grade range")
st.plotly_chart(build_prediction_interval_chart(recommendations, top_n), use_container_width=True)

st.subheader("Additional recommendation insights")
st.plotly_chart(build_score_driver_chart(recommendations, top_n), use_container_width=True)

st.subheader("Why these courses?")
for row in recommendations.head(6).itertuples():
    with st.container(border=True):
        st.markdown(f"**{row.course_number} - {row.course_title}**")
        st.write(row.explanation)
        st.caption(build_prediction_line(row))
        st.caption(build_evidence_line(row))
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
