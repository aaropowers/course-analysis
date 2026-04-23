from __future__ import annotations

import pandas as pd

from .audit import audit_degree_progress, get_eligible_courses
from .features import build_bundle_course_evidence
from .recommender import recommend_courses
from .utils import load_coursework_records, term_sort_key


TERM_SEQUENCE = ["SPRING", "FALL"]
TERM_LABELS = {"SPRING": "Spring", "SUMMER": "Summer", "FALL": "Fall"}


def _parse_coreqs(value: str) -> list[str]:
    if not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_semester_plan(recommendations: pd.DataFrame, max_credits: int = 12) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    sort_columns = ["requirement_priority", "interest_match", "score", "credits"]
    ascending = [False, False, False, True]
    if "degree_requirement_fill" in recommendations.columns:
        sort_columns = ["degree_requirement_fill", *sort_columns]
        ascending = [False, *ascending]

    working = recommendations.sort_values(
        sort_columns,
        ascending=ascending,
    )

    selected = []
    selected_courses: set[str] = set()
    used_credits = 0
    lab_selected = False
    recommendation_lookup = {row["course_number"]: row.to_dict() for _, row in working.iterrows()}

    for _, row in working.iterrows():
        if row["course_number"] in selected_courses:
            continue

        credits = int(row["credits"])
        is_lab = str(row["course_number"]).endswith("L")
        required_coreqs = [course for course in _parse_coreqs(row.get("missing_coreqs", "")) if course not in selected_courses]

        coreq_rows = []
        extra_credits = credits
        for coreq_course in required_coreqs:
            coreq_row = recommendation_lookup.get(coreq_course)
            if coreq_row is None:
                coreq_rows = []
                extra_credits = max_credits + 1
                break
            coreq_rows.append(coreq_row)
            extra_credits += int(coreq_row["credits"])

        if used_credits + credits > max_credits:
            continue
        if used_credits + extra_credits > max_credits:
            continue
        if is_lab and lab_selected:
            continue

        for coreq_row in coreq_rows:
            coreq_is_lab = str(coreq_row["course_number"]).endswith("L")
            if coreq_row["course_number"] in selected_courses:
                continue
            if coreq_is_lab and lab_selected:
                continue
            selected.append(coreq_row)
            selected_courses.add(coreq_row["course_number"])
            used_credits += int(coreq_row["credits"])
            if coreq_is_lab:
                lab_selected = True

        selected.append(row.to_dict())
        selected_courses.add(row["course_number"])
        used_credits += credits
        if is_lab:
            lab_selected = True
        if used_credits >= max_credits - 1:
            break

    plan = pd.DataFrame(selected)
    if plan.empty:
        return plan

    selected_lookup = set(plan["course_number"])
    plan["bundle_reason"] = plan.apply(
        lambda row: (
            "Corequisite-supported core"
            if str(row.get("missing_coreqs", "")).strip()
            else "Core progress"
            if row["requirement_priority"] > 0
            else "Interest fit"
            if row["interest_match"] > 0
            else "Unlocked elective"
        ),
        axis=1,
    )
    plan["bundle_coreq_ready"] = plan["missing_coreqs"].apply(
        lambda value: "Yes" if all(course in selected_lookup for course in _parse_coreqs(value)) else "No"
    )

    coursework_df = load_coursework_records("bootstrapped")
    bundle_evidence = build_bundle_course_evidence(coursework_df, plan["course_number"].tolist())
    plan = plan.merge(bundle_evidence, on="course_number", how="left")
    return plan.reset_index(drop=True)


def _next_degree_semester(transcript_df: pd.DataFrame) -> int:
    audit_result = audit_degree_progress(transcript_df)
    requirements = audit_result["requirements"]
    active = requirements[requirements["status"].isin(["Completed", "In Progress"])].copy()
    if active.empty:
        return 1

    from .utils import load_degree_plan

    semester_lookup = load_degree_plan().set_index("requirement_id")["recommended_semester"].to_dict()
    completed_semesters = active["requirement_id"].map(semester_lookup).dropna()
    if completed_semesters.empty:
        return 1
    return min(8, int(completed_semesters.max()) + 1)


def _remaining_degree_courses(transcript_df: pd.DataFrame) -> set[str]:
    from .utils import load_degree_plan, normalize_course_number, split_allowed_courses

    audit_result = audit_degree_progress(transcript_df)
    remaining_requirements = audit_result["requirements"].query("status == 'Remaining'")
    remaining_courses: set[str] = set()
    completed_courses = set(audit_result["completed_courses"]) | set(audit_result.get("in_progress_courses", []))
    transcript_courses = {
        normalize_course_number(course)
        for course in transcript_df.get("course_number", pd.Series(dtype=str)).dropna().tolist()
    }
    completed_courses |= transcript_courses

    degree_plan = load_degree_plan()
    for _, requirement in remaining_requirements.iterrows():
        plan_row = degree_plan.loc[degree_plan["requirement_id"] == requirement["requirement_id"]]
        if plan_row.empty:
            continue
        for course in split_allowed_courses(plan_row.iloc[0]["allowed_courses"]):
            normalized = normalize_course_number(course)
            if normalized and normalized not in completed_courses:
                remaining_courses.add(normalized)
    return remaining_courses


def _format_term_label(term: str) -> str:
    try:
        year, season = str(term).split("-", maxsplit=1)
    except ValueError:
        return str(term)
    return f"{TERM_LABELS.get(season.upper(), season.title())} {year}"


def _last_transcript_term(transcript_df: pd.DataFrame) -> str | None:
    if "term" not in transcript_df.columns:
        return None
    terms = [str(term).strip() for term in transcript_df["term"].dropna().tolist() if "-" in str(term)]
    if not terms:
        return None
    return sorted(terms, key=term_sort_key)[-1]


def _next_calendar_term(term: str | None) -> str:
    if not term or "-" not in str(term):
        return "2024-FALL"
    year, season = term_sort_key(str(term))
    season_name = str(term).split("-", maxsplit=1)[1].upper()
    try:
        season_index = TERM_SEQUENCE.index(season_name)
    except ValueError:
        season_index = TERM_SEQUENCE.index("FALL")
    next_index = (season_index + 1) % len(TERM_SEQUENCE)
    next_year = year + 1 if TERM_SEQUENCE[next_index] == "SPRING" else year
    return f"{next_year}-{TERM_SEQUENCE[next_index]}"


def _future_terms(transcript_df: pd.DataFrame, count: int) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    current = _last_transcript_term(transcript_df)
    for _ in range(count):
        current = _next_calendar_term(current)
        year, season_order = term_sort_key(current)
        terms.append(
            {
                "planned_term": current,
                "planned_term_label": _format_term_label(current),
                "planned_term_order": year * 10 + season_order,
            }
        )
    return terms


def _planned_transcript_rows(plan_df: pd.DataFrame, planned_term: str) -> pd.DataFrame:
    if plan_df.empty:
        return pd.DataFrame()
    rows = plan_df[["course_number", "course_title", "credits"]].copy()
    rows = rows.rename(columns={"course_title": "title", "credits": "credit_hours"})
    rows["grade"] = "IP"
    rows["status"] = "completed"
    rows["term"] = planned_term
    rows["source_type"] = "planned"
    return rows


def _attach_degree_categories(plan_df: pd.DataFrame) -> pd.DataFrame:
    if plan_df.empty:
        return plan_df
    from .utils import load_degree_plan, split_allowed_courses

    category_lookup: dict[str, str] = {}
    for _, row in load_degree_plan().iterrows():
        for course in split_allowed_courses(row["allowed_courses"]):
            category_lookup.setdefault(course, row["category"])

    result = plan_df.copy()
    result["category"] = result["course_number"].map(category_lookup).fillna("Other")
    return result


def _prepare_planning_transcript(transcript_df: pd.DataFrame) -> pd.DataFrame:
    prepared = transcript_df.copy()
    if prepared.empty:
        return prepared
    if "status" not in prepared.columns:
        prepared["status"] = "completed"
    else:
        prepared["status"] = prepared["status"].fillna("completed")
        prepared.loc[prepared["status"].astype(str).str.lower().isin(["", "nan", "none"]), "status"] = "completed"
    return prepared


def build_remaining_semester_plan(
    student_profile: dict,
    max_credits: int = 12,
    final_semester: int = 8,
) -> pd.DataFrame:
    transcript_df = _prepare_planning_transcript(student_profile["transcript_df"])
    simulated_transcript = transcript_df.copy()
    start_semester = _next_degree_semester(simulated_transcript)
    planned_terms: list[pd.DataFrame] = []
    future_terms = _future_terms(transcript_df, max(0, int(final_semester) - start_semester + 1))

    for offset, planned_semester in enumerate(range(start_semester, int(final_semester) + 1)):
        remaining_courses = _remaining_degree_courses(simulated_transcript)
        if not remaining_courses:
            break
        eligible_df = get_eligible_courses(simulated_transcript)
        if eligible_df.empty:
            continue

        term_profile = {
            **student_profile,
            "transcript_df": simulated_transcript,
            "target_credit_load": max_credits,
        }
        recommendations = recommend_courses(term_profile, eligible_courses=eligible_df)
        if recommendations.empty:
            continue
        recommendations = recommendations.copy()
        recommendations["degree_requirement_fill"] = recommendations["course_number"].isin(remaining_courses)

        prioritized = recommendations.sort_values(
            ["degree_requirement_fill", "requirement_priority", "interest_match", "score", "credits"],
            ascending=[False, False, False, False, True],
        )
        plan_df = build_semester_plan(prioritized, max_credits=max_credits)
        if plan_df.empty:
            continue

        term_meta = future_terms[offset] if offset < len(future_terms) else {}
        plan_df["planned_semester"] = planned_semester
        plan_df["planned_term"] = term_meta.get("planned_term", f"PLAN-{planned_semester}")
        plan_df["planned_term_label"] = term_meta.get("planned_term_label", f"Semester {planned_semester}")
        plan_df["planned_term_order"] = term_meta.get("planned_term_order", planned_semester)
        plan_df["degree_requirement_fill"] = plan_df["course_number"].isin(remaining_courses)
        plan_df["planner_status"] = (
            "Recommended next semester"
            if planned_semester == start_semester
            else "Recommended future semester"
        )
        planned_terms.append(plan_df)

        simulated_rows = _planned_transcript_rows(plan_df, str(plan_df["planned_term"].iloc[0]))
        simulated_transcript = pd.concat([simulated_transcript, simulated_rows], ignore_index=True)

    if not planned_terms:
        return pd.DataFrame(
            columns=[
                "planned_semester",
                "planned_term",
                "planned_term_label",
                "planned_term_order",
                "course_number",
                "course_title",
                "credits",
                "score",
                "predicted_gpa",
                "eligibility_status",
                "missing_coreqs",
                "planner_status",
                "bundle_reason",
            ]
        )

    return _attach_degree_categories(pd.concat(planned_terms, ignore_index=True))
