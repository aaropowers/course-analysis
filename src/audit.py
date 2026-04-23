from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .utils import (
    build_course_lookup,
    load_catalog,
    load_degree_plan,
    load_prereqs,
    normalize_course_number,
    split_allowed_courses,
    term_sort_key,
)


@dataclass
class RequirementStatus:
    requirement_id: str
    category: str
    requirement_name: str
    status: str
    matched_course: str | None


_DONE_STATUSES = {"completed", "credit_by_exam", "transfer"}
_STATUS_PRIORITY = {
    "credit_by_exam": 4,
    "completed": 3,
    "transfer": 2,
    "in_progress": 1,
}


def _normalize_transcript_status(value: object) -> str:
    status = str(value or "").strip().lower()
    if status in {"completed", "in_progress", "credit_by_exam", "transfer"}:
        return status
    if status in {"in progress", "inprogress", "current"}:
        return "in_progress"
    if status in {"credit by exam", "credit-by-exam", "cbe"}:
        return "credit_by_exam"
    return status or "completed"


def _transcript_status_map(transcript_df: pd.DataFrame) -> dict[str, str]:
    """Map each course on the transcript to its best-available status.

    When a course appears multiple times (e.g., RHE306 as both ``transfer`` and
    ``credit_by_exam``), the higher-priority status wins so downstream logic
    treats the institutional record as authoritative.
    """
    status_by_course: dict[str, str] = {}
    if transcript_df is None or transcript_df.empty:
        return status_by_course
    has_status_col = "status" in transcript_df.columns
    for _, row in transcript_df.iterrows():
        course = normalize_course_number(row.get("course_number"))
        if not course:
            continue
        raw_status = row.get("status") if has_status_col else None
        status = _normalize_transcript_status(raw_status)
        existing = status_by_course.get(course)
        if existing is None or _STATUS_PRIORITY.get(status, 0) > _STATUS_PRIORITY.get(existing, 0):
            status_by_course[course] = status
    return status_by_course


def _completed_courses(transcript_df: pd.DataFrame) -> set[str]:
    status_map = _transcript_status_map(transcript_df)
    return {course for course, status in status_map.items() if status in _DONE_STATUSES}


def _in_progress_courses(transcript_df: pd.DataFrame) -> set[str]:
    status_map = _transcript_status_map(transcript_df)
    return {course for course, status in status_map.items() if status == "in_progress"}


def audit_degree_progress(transcript_df: pd.DataFrame, degree_plan_df: pd.DataFrame | None = None) -> dict:
    degree_plan_df = degree_plan_df if degree_plan_df is not None else load_degree_plan()
    completed_courses = _completed_courses(transcript_df)
    in_progress_courses = _in_progress_courses(transcript_df)

    requirement_rows: list[RequirementStatus] = []
    completed_hours = 0
    in_progress_hours = 0
    total_hours = int(degree_plan_df["required_hours"].sum())

    for _, row in degree_plan_df.iterrows():
        allowed = split_allowed_courses(row["allowed_courses"])
        completed_matches = sorted(completed_courses & allowed)
        in_progress_matches = sorted(in_progress_courses & allowed)

        if completed_matches:
            matched_course = completed_matches[0]
            status = "Completed"
            completed_hours += int(row["required_hours"])
        elif in_progress_matches:
            matched_course = in_progress_matches[0]
            status = "In Progress"
            in_progress_hours += int(row["required_hours"])
        else:
            matched_course = None
            status = "Remaining"

        requirement_rows.append(
            RequirementStatus(
                requirement_id=row["requirement_id"],
                category=row["category"],
                requirement_name=row["requirement_name"],
                status=status,
                matched_course=matched_course,
            )
        )

    requirement_df = pd.DataFrame([item.__dict__ for item in requirement_rows])
    summary = (
        requirement_df.groupby(["category", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    completed_req_count = int((requirement_df["status"] == "Completed").sum())
    in_progress_req_count = int((requirement_df["status"] == "In Progress").sum())
    total_req_count = int(len(requirement_df))

    return {
        "requirements": requirement_df,
        "summary": summary,
        "completed_courses": sorted(completed_courses),
        "in_progress_courses": sorted(in_progress_courses),
        "completed_hours": completed_hours,
        "in_progress_hours": in_progress_hours,
        "total_hours": total_hours,
        "progress_percent": round((completed_req_count / total_req_count) * 100, 1) if total_req_count else 0.0,
        "in_progress_percent": round((in_progress_req_count / total_req_count) * 100, 1) if total_req_count else 0.0,
    }


def get_missing_requirements(audit_result: dict) -> pd.DataFrame:
    return audit_result["requirements"].query("status == 'Remaining'").reset_index(drop=True)


def _prereq_status_for_course(
    course_number: str,
    completed_courses: set[str],
    prereq_df: pd.DataFrame,
    planned_courses: set[str] | None = None,
) -> tuple[bool, list[str], list[str]]:
    planned_courses = planned_courses or set()
    rows = prereq_df[prereq_df["course_number"] == course_number]
    prereq_rows = rows[rows["prereq_type"] == "PREREQ"]
    coreq_rows = rows[rows["prereq_type"] == "COREQ"]

    missing_prereqs = sorted(
        set(prereq_rows.loc[~prereq_rows["prerequisite_course"].isin(completed_courses), "prerequisite_course"])
    )
    available_for_coreq = completed_courses | planned_courses | {course_number}
    missing_coreqs = sorted(
        set(coreq_rows.loc[~coreq_rows["prerequisite_course"].isin(available_for_coreq), "prerequisite_course"])
    )
    return len(missing_prereqs) == 0, missing_prereqs, missing_coreqs


def evaluate_course_dependencies(
    transcript_df: pd.DataFrame,
    prereq_df: pd.DataFrame | None = None,
    catalog_df: pd.DataFrame | None = None,
    planned_courses: set[str] | None = None,
) -> pd.DataFrame:
    prereq_df = prereq_df if prereq_df is not None else load_prereqs()
    catalog_df = catalog_df if catalog_df is not None else load_catalog()
    completed_courses = _completed_courses(transcript_df)
    in_progress_courses = _in_progress_courses(transcript_df)
    planned_courses = {normalize_course_number(course) for course in (planned_courses or set())}
    coreq_satisfiers = planned_courses | in_progress_courses

    rows = []
    for _, course in catalog_df.iterrows():
        course_number = course["course_number"]
        is_completed = course_number in completed_courses
        is_in_progress = course_number in in_progress_courses
        has_prereqs, missing_prereqs, missing_coreqs = _prereq_status_for_course(
            course_number,
            completed_courses,
            prereq_df,
            planned_courses=coreq_satisfiers,
        )

        if is_completed:
            eligibility_status = "Completed"
        elif is_in_progress:
            eligibility_status = "In Progress"
        elif not has_prereqs:
            eligibility_status = "Locked"
        elif missing_coreqs:
            eligibility_status = "Eligible with corequisite"
        else:
            eligibility_status = "Eligible"

        rows.append(
            {
                "course_number": course_number,
                "course_title": course["course_title"],
                "credits": course["credits"],
                "elective_category": course["elective_category"],
                "recommended_semester": course["recommended_semester"],
                "is_completed": is_completed,
                "is_in_progress": is_in_progress,
                "is_eligible": eligibility_status in {"Eligible", "Eligible with corequisite"},
                "eligibility_status": eligibility_status,
                "missing_prereqs": ", ".join(missing_prereqs),
                "missing_coreqs": ", ".join(missing_coreqs),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["recommended_semester", "course_number"],
        ascending=[True, True],
    ).reset_index(drop=True)


def get_eligible_courses(
    transcript_df: pd.DataFrame,
    prereq_df: pd.DataFrame | None = None,
    catalog_df: pd.DataFrame | None = None,
    planned_courses: set[str] | None = None,
) -> pd.DataFrame:
    evaluated = evaluate_course_dependencies(
        transcript_df,
        prereq_df=prereq_df,
        catalog_df=catalog_df,
        planned_courses=planned_courses,
    )
    return evaluated[
        ~evaluated["is_completed"] & ~evaluated.get("is_in_progress", pd.Series(False, index=evaluated.index))
    ].sort_values(
        ["is_eligible", "recommended_semester", "course_number"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def get_locked_courses(transcript_df: pd.DataFrame) -> pd.DataFrame:
    eligible_df = get_eligible_courses(transcript_df)
    return eligible_df.query("eligibility_status == 'Locked'").reset_index(drop=True)


def build_degree_plan_progression(transcript_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    degree_plan_df = load_degree_plan().copy()
    catalog_df = load_catalog()
    prereq_df = load_prereqs()
    dependency_df = evaluate_course_dependencies(transcript_df, prereq_df=prereq_df, catalog_df=catalog_df)

    completed_set = set(dependency_df.loc[dependency_df["is_completed"], "course_number"])
    in_progress_set = set(dependency_df.loc[dependency_df.get("is_in_progress", False) == True, "course_number"])  # noqa: E712

    requirement_rows = []
    for _, row in degree_plan_df.iterrows():
        allowed_courses = sorted(split_allowed_courses(row["allowed_courses"]))
        completed_match = next((c for c in allowed_courses if c in completed_set), None)
        in_progress_match = (
            next((c for c in allowed_courses if c in in_progress_set), None)
            if completed_match is None
            else None
        )

        matched_course: str | None = None
        if completed_match:
            requirement_status = "Completed"
            primary_course = completed_match
            matched_course = completed_match
        elif in_progress_match:
            requirement_status = "In Progress"
            primary_course = in_progress_match
            matched_course = in_progress_match
        else:
            eligible_choices = dependency_df[
                dependency_df["course_number"].isin(allowed_courses) & dependency_df["is_eligible"]
            ]["course_number"].tolist()
            locked_choices = dependency_df[
                dependency_df["course_number"].isin(allowed_courses)
                & (dependency_df["eligibility_status"] == "Locked")
            ]["course_number"].tolist()
            if eligible_choices:
                requirement_status = "Eligible"
                primary_course = eligible_choices[0]
            elif locked_choices:
                requirement_status = "Locked"
                primary_course = locked_choices[0]
            else:
                primary_course = allowed_courses[0] if allowed_courses else ""
                requirement_status = "Locked"

        primary_row = dependency_df.loc[dependency_df["course_number"] == primary_course]
        missing_prereqs = ""
        missing_coreqs = ""
        if not primary_row.empty:
            primary_info = primary_row.iloc[0]
            missing_prereqs = primary_info.get("missing_prereqs", "") or ""
            missing_coreqs = primary_info.get("missing_coreqs", "") or ""

        requirement_rows.append(
            {
                "requirement_id": row["requirement_id"],
                "category": row["category"],
                "requirement_name": row["requirement_name"],
                "course_number": primary_course,
                "allowed_courses": row["allowed_courses"],
                "recommended_semester": row["recommended_semester"],
                "status": requirement_status,
                "matched_course": matched_course,
                "missing_prereqs": missing_prereqs,
                "missing_coreqs": missing_coreqs,
            }
        )

    requirement_df = pd.DataFrame(requirement_rows)

    plan_courses = set(requirement_df["course_number"])
    edge_df = prereq_df[
        prereq_df["course_number"].isin(plan_courses) & prereq_df["prerequisite_course"].isin(plan_courses)
    ].copy()
    return requirement_df, edge_df.reset_index(drop=True)


_TRANSCRIPT_PROGRESSION_COLUMNS = [
    "node_id",
    "course_number",
    "course_title",
    "term",
    "term_label",
    "term_order",
    "status",
    "credit_hours",
    "grade",
    "source_type",
    "category",
]


def _format_term_label(term: str) -> str:
    term = str(term or "").strip()
    if "-" not in term:
        return term
    year_text, season = term.split("-", maxsplit=1)
    season_title = season.strip().title() if season else ""
    year_text = year_text.strip()
    if season_title and year_text:
        return f"{season_title} {year_text}"
    return term


def _term_order(term: str) -> int:
    try:
        year, season_order = term_sort_key(term)
    except Exception:
        return 10**9
    return year * 10 + season_order


def build_transcript_progression(transcript_df: pd.DataFrame) -> pd.DataFrame:
    """Build a term-by-term timeline of the courses actually on the transcript.

    Unlike :func:`build_degree_plan_progression`, which is anchored to degree
    requirements, this builder emits one node per transcript row grouped by the
    term the course was taken (e.g., Fall 2022, Spring 2023). It powers the
    "Transcript View" of the progression map on the Degree Audit page and
    intentionally includes every course -- completed, credit-by-exam, transfer,
    and in-progress -- so the timeline mirrors real academic history.
    """
    if transcript_df is None or transcript_df.empty:
        return pd.DataFrame(columns=_TRANSCRIPT_PROGRESSION_COLUMNS)

    rows = transcript_df.copy()
    rows["course_number"] = rows["course_number"].map(normalize_course_number)

    catalog_lookup = load_catalog().set_index("course_number").to_dict(orient="index")
    plan_category_lookup: dict[str, str] = {}
    for _, plan_row in load_degree_plan().iterrows():
        for code in split_allowed_courses(plan_row["allowed_courses"]):
            plan_category_lookup.setdefault(code, plan_row["category"])

    records: list[dict] = []
    for idx, row in rows.reset_index(drop=True).iterrows():
        course_number = row.get("course_number") or ""
        term = str(row.get("term") or "").strip()
        if not course_number or not term or "-" not in term:
            continue

        title = str(row.get("title") or "").strip()
        if not title:
            title = catalog_lookup.get(course_number, {}).get("course_title", "")

        credit_hours = pd.to_numeric(row.get("credit_hours"), errors="coerce")
        credit_hours = float(credit_hours) if pd.notna(credit_hours) else None

        grade_raw = row.get("grade")
        grade = None if grade_raw is None or pd.isna(grade_raw) else str(grade_raw)

        source_type = str(row.get("source_type") or "").strip()
        status = _normalize_transcript_status(row.get("status"))
        category = plan_category_lookup.get(course_number, "Other")

        records.append(
            {
                "node_id": f"{term}::{course_number}::{idx}",
                "course_number": course_number,
                "course_title": title,
                "term": term,
                "term_label": _format_term_label(term),
                "term_order": _term_order(term),
                "status": status,
                "credit_hours": credit_hours,
                "grade": grade,
                "source_type": source_type,
                "category": category,
            }
        )

    if not records:
        return pd.DataFrame(columns=_TRANSCRIPT_PROGRESSION_COLUMNS)

    result = pd.DataFrame(records, columns=_TRANSCRIPT_PROGRESSION_COLUMNS)
    category_order = {
        "Math": 1,
        "Science": 2,
        "Computing": 3,
        "General Education": 4,
        "ME Core": 5,
        "ME Lab": 6,
        "Gateway Elective": 7,
        "Other": 8,
    }
    result["_category_rank"] = result["category"].map(category_order).fillna(99)
    result = (
        result.sort_values(["term_order", "_category_rank", "course_number"])
        .drop(columns="_category_rank")
        .reset_index(drop=True)
    )
    return result


def annotate_courses(courses: list[str]) -> list[str]:
    lookup = build_course_lookup()
    labeled = []
    for course in courses:
        meta = lookup.get(course, {})
        title = meta.get("course_title", "")
        labeled.append(f"{course} - {title}" if title else course)
    return labeled
