from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import (
    build_coenrollment_effect,
    build_coenrollment_features,
    build_course_grade_distribution,
    build_department_gpa_heatmap,
    compute_course_similarity,
    get_top_courses_by_enrollment,
)
from src.utils import load_catalog, load_coursework_records


st.title("Insights")
st.caption("Coursework-backed analytics powered by the expanded bootstrapped dataset.")

catalog_df = load_catalog()
coursework_df = load_coursework_records("bootstrapped")
coenrollment_df = build_coenrollment_features(coursework_df)

default_courses = get_top_courses_by_enrollment(coursework_df, n=25, graded_only=True)
if not default_courses:
    st.warning("No coursework-backed analytics are available yet.")
    st.stop()

course_labels = {
    row.course_number: f"{row.course_number} - {row.course_title}"
    for row in catalog_df.drop_duplicates("course_number").itertuples()
}

st.subheader("Course Grade Distribution")
selected_course = st.selectbox(
    "Course for histogram",
    options=default_courses,
    index=0,
    format_func=lambda course: course_labels.get(course, course),
)

distribution_df, distribution_summary = build_course_grade_distribution(coursework_df, selected_course)
if distribution_df.empty:
    st.info(f"No letter-grade data is available for {selected_course}.")
else:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Sample size", distribution_summary["sample_size"])
    metric_cols[1].metric("Average GPA", distribution_summary["avg_gpa"] if distribution_summary["avg_gpa"] is not None else "N/A")
    metric_cols[2].metric("Pass rate", f"{distribution_summary['pass_rate']}%" if distribution_summary["pass_rate"] is not None else "N/A")

    histogram_fig = go.Figure()
    histogram_fig.add_trace(
        go.Bar(
            x=distribution_df["grade"],
            y=distribution_df["count"],
            name="Grade count",
            marker_color="#2B6CB0",
        )
    )
    histogram_fig.add_trace(
        go.Scatter(
            x=distribution_df["grade"],
            y=distribution_df["cumulative_pct"],
            mode="lines+markers",
            name="Cumulative %",
            yaxis="y2",
            line={"color": "#DD6B20", "width": 3},
        )
    )
    histogram_fig.update_layout(
        title=f"Grade distribution for {selected_course}",
        xaxis_title="Letter grade",
        yaxis_title="Student count",
        yaxis2={"title": "Cumulative %", "overlaying": "y", "side": "right", "range": [0, 100]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(histogram_fig, use_container_width=True)

st.subheader("Co-Enrollment Grade Effect")
comparison_cols = st.columns(2)
target_course = comparison_cols[0].selectbox(
    "Target course",
    options=default_courses,
    index=min(1, len(default_courses) - 1),
    format_func=lambda course: course_labels.get(course, course),
    key="target_course",
)
companion_options = [course for course in default_courses if course != target_course]
companion_course = comparison_cols[1].selectbox(
    "Companion course",
    options=companion_options,
    index=0,
    format_func=lambda course: course_labels.get(course, course),
)

effect_payload = build_coenrollment_effect(coursework_df, target_course, companion_course)
effect_df = effect_payload["comparison_df"]
effect_summary = effect_payload["summary"]

summary_cols = st.columns(3)
summary_cols[0].metric("With companion", effect_summary["with_count"])
summary_cols[1].metric("Without companion", effect_summary["without_count"])
summary_cols[2].metric("GPA delta", effect_summary["gpa_delta"] if effect_summary["gpa_delta"] is not None else "N/A")

if effect_summary["with_count"] < 10 or effect_summary["without_count"] < 10:
    st.info("This course pairing is sparse in the current dataset. Try a more common companion course for a clearer comparison.")
elif effect_df.empty:
    st.info("No overlapping letter-grade data is available for this course pairing.")
else:
    effect_fig = px.bar(
        effect_df,
        x="grade",
        y="count",
        color="group",
        barmode="group",
        category_orders={"grade": ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]},
        title=f"How {companion_course} changes the grade distribution in {target_course}",
    )
    st.plotly_chart(effect_fig, use_container_width=True)
    avg_cols = st.columns(2)
    avg_cols[0].metric("Avg GPA with companion", effect_summary["with_avg_gpa"] if effect_summary["with_avg_gpa"] is not None else "N/A")
    avg_cols[1].metric("Avg GPA without companion", effect_summary["without_avg_gpa"] if effect_summary["without_avg_gpa"] is not None else "N/A")

st.subheader("Department GPA Heatmap")
top_department_count = st.slider("Departments to include", min_value=6, max_value=20, value=12, step=1)
heatmap_data = build_department_gpa_heatmap(coursework_df, top_n=top_department_count)
if heatmap_data.empty:
    st.info("No department GPA data is available.")
else:
    heatmap_fig = px.imshow(
        heatmap_data,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        origin="lower",
        title="Average GPA by department and semester",
    )
    heatmap_fig.update_layout(coloraxis_colorbar_title="Avg GPA")
    st.plotly_chart(heatmap_fig, use_container_width=True)

st.subheader("Top Co-Enrollment Pairs")
pair_limit = st.slider("Number of pairs to show", min_value=5, max_value=25, value=12, step=1)
if coenrollment_df.empty:
    st.info("No co-enrollment pairs are available.")
else:
    top_pairs = coenrollment_df.head(pair_limit).copy()
    top_pairs["pair"] = top_pairs["course_a"] + " + " + top_pairs["course_b"]
    pair_fig = px.bar(
        top_pairs.sort_values("coenrollment_count", ascending=True),
        x="coenrollment_count",
        y="pair",
        orientation="h",
        title="Most common same-term course pairs",
    )
    st.plotly_chart(pair_fig, use_container_width=True)
    st.dataframe(top_pairs[["course_a", "course_b", "coenrollment_count"]], use_container_width=True, hide_index=True)

with st.expander("Course Similarity Inspector", expanded=False):
    similarity_course = st.selectbox(
        "Inspect similar courses",
        options=default_courses,
        index=0,
        format_func=lambda course: course_labels.get(course, course),
        key="similarity_course",
    )
    similarity = compute_course_similarity(catalog_df)
    top_similar = (
        similarity.loc[similarity_course]
        .drop(index=similarity_course)
        .sort_values(ascending=False)
        .head(5)
        .rename("similarity_score")
        .reset_index()
        .rename(columns={"index": "course_number"})
        .merge(catalog_df[["course_number", "course_title", "avg_gpa"]], on="course_number", how="left")
    )
    st.dataframe(top_similar, use_container_width=True, hide_index=True)
