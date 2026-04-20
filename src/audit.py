from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .utils import build_course_lookup, load_catalog, load_degree_plan, load_prereqs, normalize_course_number, split_allowed_courses


@dataclass
class RequirementStatus:
    requirement_id: str
    category: str
    requirement_name: str
    status: str
    matched_course: str | None


def _completed_courses(transcript_df: pd.DataFrame) -> set[str]:
    return {normalize_course_number(course) for course in transcript_df["course_number"].tolist()}


def audit_degree_progress(transcript_df: pd.DataFrame, degree_plan_df: pd.DataFrame | None = None) -> dict:
    degree_plan_df = degree_plan_df if degree_plan_df is not None else load_degree_plan()
    completed_courses = _completed_courses(transcript_df)

    requirement_rows: list[RequirementStatus] = []
    completed_hours = 0
    total_hours = int(degree_plan_df["required_hours"].sum())

    for _, row in degree_plan_df.iterrows():
        allowed = split_allowed_courses(row["allowed_courses"])
        matches = sorted(completed_courses & allowed)
        matched_course = matches[0] if matches else None
        status = "Completed" if matched_course else "Remaining"
        if matched_course:
            completed_hours += int(row["required_hours"])

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
    total_req_count = int(len(requirement_df))

    return {
        "requirements": requirement_df,
        "summary": summary,
        "completed_courses": sorted(completed_courses),
        "completed_hours": completed_hours,
        "total_hours": total_hours,
        "progress_percent": round((completed_req_count / total_req_count) * 100, 1) if total_req_count else 0.0,
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
    planned_courses = {normalize_course_number(course) for course in (planned_courses or set())}

    rows = []
    for _, course in catalog_df.iterrows():
        course_number = course["course_number"]
        is_completed = course_number in completed_courses
        has_prereqs, missing_prereqs, missing_coreqs = _prereq_status_for_course(
            course_number,
            completed_courses,
            prereq_df,
            planned_courses=planned_courses,
        )

        if is_completed:
            eligibility_status = "Completed"
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
        ~evaluated["is_completed"]
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

    course_meta = catalog_df[
        ["course_number", "course_title", "recommended_semester", "elective_category"]
    ].drop_duplicates()
    requirement_rows = []
    for _, row in degree_plan_df.iterrows():
        allowed_courses = sorted(split_allowed_courses(row["allowed_courses"]))
        matched_course = next(
            (course for course in allowed_courses if course in set(dependency_df.loc[dependency_df["is_completed"], "course_number"])),
            None,
        )

        if len(allowed_courses) == 1:
            primary_course = allowed_courses[0]
            course_info = dependency_df.loc[dependency_df["course_number"] == primary_course].iloc[0].to_dict()
        else:
            eligible_choices = dependency_df[
                dependency_df["course_number"].isin(allowed_courses) & dependency_df["is_eligible"]
            ]["course_number"].tolist()
            locked_choices = dependency_df[
                dependency_df["course_number"].isin(allowed_courses) & (dependency_df["eligibility_status"] == "Locked")
            ]["course_number"].tolist()
            if matched_course:
                requirement_status = "Completed"
            elif eligible_choices:
                requirement_status = "Eligible"
            else:
                requirement_status = "Locked"
            primary_course = matched_course or (eligible_choices[0] if eligible_choices else allowed_courses[0])
            course_info = {
                "eligibility_status": requirement_status,
                "missing_prereqs": "",
                "missing_coreqs": "",
                "course_title": row["requirement_name"],
                "course_number": primary_course,
            }
            if not matched_course and not eligible_choices and locked_choices:
                first_locked = dependency_df.loc[dependency_df["course_number"] == locked_choices[0]].iloc[0]
                course_info["missing_prereqs"] = first_locked["missing_prereqs"]
                course_info["missing_coreqs"] = first_locked["missing_coreqs"]

        requirement_rows.append(
            {
                "requirement_id": row["requirement_id"],
                "category": row["category"],
                "requirement_name": row["requirement_name"],
                "course_number": primary_course,
                "allowed_courses": row["allowed_courses"],
                "recommended_semester": row["recommended_semester"],
                "status": "Completed" if matched_course else course_info["eligibility_status"],
                "matched_course": matched_course,
                "missing_prereqs": course_info.get("missing_prereqs", ""),
                "missing_coreqs": course_info.get("missing_coreqs", ""),
            }
        )

    requirement_df = pd.DataFrame(requirement_rows)

    plan_courses = set(requirement_df["course_number"])
    edge_df = prereq_df[
        prereq_df["course_number"].isin(plan_courses) & prereq_df["prerequisite_course"].isin(plan_courses)
    ].copy()
    return requirement_df, edge_df.reset_index(drop=True)


def annotate_courses(courses: list[str]) -> list[str]:
    lookup = build_course_lookup()
    labeled = []
    for course in courses:
        meta = lookup.get(course, {})
        title = meta.get("course_title", "")
        labeled.append(f"{course} - {title}" if title else course)
    return labeled
