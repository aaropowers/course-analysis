from __future__ import annotations

import pandas as pd

from .features import build_bundle_course_evidence
from .utils import load_coursework_records


def _parse_coreqs(value: str) -> list[str]:
    if not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_semester_plan(recommendations: pd.DataFrame, max_credits: int = 12) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    working = recommendations.sort_values(
        ["requirement_priority", "interest_match", "score", "credits"],
        ascending=[False, False, False, True],
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
