from __future__ import annotations

import pandas as pd

from .utils import GRADE_POINTS, load_catalog, normalize_course_number


def parse_transcript(input_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    working = input_df.copy()
    working.columns = [column.strip().lower() for column in working.columns]

    if "course_number" not in working.columns:
        raise ValueError("Transcript input must include a 'course_number' column.")

    if "grade" not in working.columns:
        working["grade"] = None
    if "term" not in working.columns:
        working["term"] = None

    working["course_number"] = working["course_number"].map(normalize_course_number)
    working["grade"] = working["grade"].astype(str).str.upper().replace({"NAN": None, "NONE": None})
    working["grade_points"] = working["grade"].map(GRADE_POINTS)

    known_courses = set(load_catalog()["course_number"])
    invalid_courses = sorted({course for course in working["course_number"] if course and course not in known_courses})

    parsed = working[working["course_number"].isin(known_courses)].drop_duplicates(subset=["course_number"]).reset_index(drop=True)
    return parsed, invalid_courses


def build_transcript_from_course_list(course_numbers: list[str], grades: dict[str, str] | None = None) -> pd.DataFrame:
    grades = grades or {}
    rows = []
    for course_number in course_numbers:
        normalized = normalize_course_number(course_number)
        rows.append(
            {
                "course_number": normalized,
                "grade": grades.get(normalized),
                "term": None,
                "grade_points": GRADE_POINTS.get(grades.get(normalized)),
            }
        )
    return pd.DataFrame(rows)
