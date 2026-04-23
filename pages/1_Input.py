from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from src.cleaning import build_transcript_from_course_list, parse_transcript
from src.transcript_pdf import parse_ut_transcript_pdf, to_transcript_dataframe
from src.utils import load_catalog, load_transcripts, normalize_course_number, transcript_gpa


def _session_transcript_from_pdf(parsed_full_df: pd.DataFrame, parsed_model_df: pd.DataFrame) -> pd.DataFrame:
    """Keep catalog-valid courses like parse_transcript, but preserve PDF metadata columns.

    When the same course_number appears more than once (e.g. transfer then credit-by-exam),
    prefer the institutional row (credit_by_exam / completed) over transfer so hours and
    GPA align with UT totals.
    """
    if parsed_model_df.empty:
        return pd.DataFrame()
    full = parsed_full_df.copy()
    full["course_number"] = full["course_number"].map(normalize_course_number)
    valid_codes = set(parsed_model_df["course_number"])

    status_priority = {"credit_by_exam": 3, "completed": 2, "in_progress": 1, "transfer": 0}

    session_rows: list[dict] = []
    for code in parsed_model_df["course_number"].drop_duplicates():
        if code not in valid_codes:
            continue
        hits = full[full["course_number"] == code]
        if hits.empty:
            continue
        if "status" in hits.columns:
            scored = hits.copy()
            scored["_prio"] = scored["status"].astype(str).str.lower().map(
                lambda s: status_priority.get(s, 0)
            )
            ch = pd.to_numeric(scored.get("credit_hours", pd.Series(0, index=scored.index)), errors="coerce").fillna(0)
            scored = scored.assign(_ch=ch).sort_values(["_prio", "_ch"], ascending=[False, False])
            chosen = scored.iloc[0].drop(labels=["_prio", "_ch"], errors="ignore")
            session_rows.append(chosen.to_dict())
        else:
            session_rows.append(hits.iloc[0].to_dict())
    return pd.DataFrame(session_rows).reset_index(drop=True)


def _save_student_gpa(transcript_df: pd.DataFrame) -> None:
    computed_gpa = transcript_gpa(transcript_df)
    if computed_gpa is not None:
        st.session_state["student_gpa"] = computed_gpa
    else:
        st.session_state.pop("student_gpa", None)


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
        transcript = demo_df[demo_df["student_id"] == student_id].copy()
        st.session_state["transcript_df"] = transcript
        st.session_state["active_student_id"] = student_id
        _save_student_gpa(transcript)
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
        _save_student_gpa(parsed_df)
        st.success(f"Loaded {len(parsed_df)} valid completed courses from upload.")
        if invalid_courses:
            st.warning("Ignored unknown courses: " + ", ".join(invalid_courses))

with st.expander("Upload unofficial transcript PDF (UT format)", expanded=True):
    st.caption("Upload an Academic Summary PDF to parse courses, grades, and terms.")
    uploaded_pdf = st.file_uploader("Upload unofficial transcript", type=["pdf"], key="transcript_pdf_uploader")
    if uploaded_pdf is not None:
        parse_result = parse_ut_transcript_pdf(uploaded_pdf)
        parsed_full_df = to_transcript_dataframe(parse_result)
        parsed_model_df, invalid_courses = parse_transcript(parsed_full_df[["course_number", "grade", "term"]].copy())

        st.markdown(
            f"**Parsed rows:** {len(parsed_full_df)} | **Valid catalog matches:** {len(parsed_model_df)} | "
            f"**Unknown courses:** {len(invalid_courses)}"
        )

        if parse_result.warnings:
            with st.expander(f"Parser warnings ({len(parse_result.warnings)})", expanded=False):
                st.write(parse_result.warnings)

        if invalid_courses:
            with st.expander("Courses not in current app catalog", expanded=False):
                st.write(invalid_courses)

        preview_cols = [col for col in ["course_number", "title", "grade", "term", "status"] if col in parsed_full_df.columns]
        st.dataframe(parsed_full_df[preview_cols].fillna(""), use_container_width=True, hide_index=True)

        full_buffer = io.StringIO()
        parsed_full_df.to_csv(full_buffer, index=False)
        st.download_button(
            "Export CSV (full parsed output)",
            data=full_buffer.getvalue(),
            file_name=f"{uploaded_pdf.name.rsplit('.', maxsplit=1)[0]}_parsed_full.csv",
            mime="text/csv",
            key="download_full_parsed_csv",
        )

        model_buffer = io.StringIO()
        parsed_model_df.to_csv(model_buffer, index=False)
        st.download_button(
            "Export CSV (model input format)",
            data=model_buffer.getvalue(),
            file_name=f"{uploaded_pdf.name.rsplit('.', maxsplit=1)[0]}_model_input.csv",
            mime="text/csv",
            key="download_model_input_csv",
        )

        if st.button("Use parsed transcript from PDF", type="primary"):
            session_df = _session_transcript_from_pdf(parsed_full_df, parsed_model_df)
            st.session_state["transcript_df"] = session_df
            st.session_state["active_student_id"] = "pdf_uploaded_student"
            st.session_state["invalid_courses"] = invalid_courses
            _save_student_gpa(session_df)
            st.success(f"Loaded {len(session_df)} valid completed courses from PDF (with transcript metadata).")

with st.expander("Manual transcript builder", expanded=False):
    selected_courses = st.multiselect(
        "Completed courses",
        options=catalog_df["course_number"].tolist(),
        default=st.session_state.get("transcript_df", pd.DataFrame()).get("course_number", pd.Series(dtype=str)).tolist(),
        format_func=lambda course: f"{course} - {catalog_df.loc[catalog_df['course_number'] == course, 'course_title'].iloc[0]}",
    )
    if st.button("Use selected courses"):
        manual_df = build_transcript_from_course_list(selected_courses)
        st.session_state["transcript_df"] = manual_df
        st.session_state["active_student_id"] = "manual_student"
        _save_student_gpa(manual_df)
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
    _save_student_gpa(current_transcript)

if not current_transcript.empty:
    current_transcript["course_number"] = current_transcript["course_number"].map(normalize_course_number)
    preview_cols = ["course_number", "grade", "term"]
    for extra in ("status", "credit_hours"):
        if extra in current_transcript.columns:
            preview_cols.append(extra)
    preview = current_transcript[[col for col in preview_cols if col in current_transcript.columns]].fillna("")
    st.subheader("Current transcript in session")
    st.dataframe(preview, use_container_width=True, hide_index=True)
