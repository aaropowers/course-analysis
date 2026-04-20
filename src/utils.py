from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_COURSEWORK_PATH = ROOT / "coursework.csv"
CLEAN_COURSEWORK_PATH = DATA_DIR / "coursework_cleaned.csv"
BOOTSTRAPPED_COURSEWORK_PATH = DATA_DIR / "coursework_bootstrapped.csv"

GRADE_POINTS = {
    "A": 4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B": 3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C": 2.0,
    "C-": 1.7,
    "D+": 1.3,
    "D": 1.0,
    "D-": 0.7,
    "F": 0.0,
    "CR": None,
}

VALID_GRADES = set(GRADE_POINTS) | {"Q", "W", "IP"}
SEMESTER_MAP = {
    "FALL": "FALL",
    "FALL ": "FALL",
    "SPRING": "SPRING",
    "SPRING ": "SPRING",
    "SPING": "SPRING",
    "SPRNG": "SPRING",
    "SUMMER": "SUMMER",
    "SUMMER ": "SUMMER",
    "SUMMMER": "SUMMER",
}
GRADE_NORMALIZE_MAP = {
    "A+": "A",
    "CREDIT": "CR",
    "IN PROGRESS": "IP",
    "TBD": "IP",
    "NOW": "IP",
    "NAN": pd.NA,
    "NONE": pd.NA,
    "-": pd.NA,
    "#": pd.NA,
    "": pd.NA,
    " ": pd.NA,
}


def normalize_course_number(course_number: str | None) -> str:
    if course_number is None:
        return ""
    value = str(course_number).upper().strip()
    for token in (" ", "-", "_"):
        value = value.replace(token, "")
    return value


def derive_department(course_number: str | None) -> str:
    value = normalize_course_number(course_number)
    match = re.match(r"([A-Z]+)", value)
    return match.group(1) if match else ""


def normalize_semester(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = SEMESTER_MAP.get(str(value).strip().upper())
    return normalized if normalized else None


def normalize_grade(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    normalized = GRADE_NORMALIZE_MAP.get(normalized, normalized)
    if pd.isna(normalized) or normalized is None:
        return None
    return normalized if normalized in VALID_GRADES else None


def _normalize_frame_course_numbers(df: pd.DataFrame, column: str = "course_number") -> pd.DataFrame:
    result = df.copy()
    if column in result.columns:
        result[column] = result[column].map(normalize_course_number)
    return result


@lru_cache(maxsize=1)
def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "course_catalog.csv")
    return _normalize_frame_course_numbers(df)


@lru_cache(maxsize=1)
def load_degree_plan() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "degree_plan.csv")
    df["required_flag"] = df["required_flag"].astype(str).str.upper().eq("TRUE")
    return df


@lru_cache(maxsize=1)
def load_prereqs() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "prereqs.csv")
    df = _normalize_frame_course_numbers(df)
    df["prerequisite_course"] = df["prerequisite_course"].map(normalize_course_number)
    df["prereq_type"] = (
        df["prereq_type"]
        .astype(str)
        .str.upper()
        .replace(
            {
                "AND": "PREREQ",
                "PREREQ": "PREREQ",
                "COREQ": "COREQ",
            }
        )
    )
    return df


@lru_cache(maxsize=1)
def load_elective_groups() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "elective_groups.csv")
    return _normalize_frame_course_numbers(df)


@lru_cache(maxsize=1)
def load_transcripts() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "synthetic_transcripts.csv")
    df = _normalize_frame_course_numbers(df)
    df["grade_points"] = df["grade"].map(GRADE_POINTS)
    return df


def _prepare_coursework_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "course_number" in result.columns:
        result["course_number"] = result["course_number"].map(normalize_course_number)
        result["department"] = result.get("department", pd.Series(index=result.index, dtype=object)).fillna(
            result["course_number"].map(derive_department)
        )
    if "semester" in result.columns:
        result["semester"] = result["semester"].map(normalize_semester).fillna(result["semester"])
    if "grade" in result.columns:
        result["grade"] = result["grade"].map(normalize_grade)
        result["grade_points"] = result["grade"].map(GRADE_POINTS)
    return result


@lru_cache(maxsize=3)
def load_coursework_records(dataset: str = "bootstrapped") -> pd.DataFrame:
    dataset = dataset.lower()
    if dataset == "raw":
        path = RAW_COURSEWORK_PATH
    elif dataset == "cleaned":
        path = CLEAN_COURSEWORK_PATH
    elif dataset == "bootstrapped":
        path = BOOTSTRAPPED_COURSEWORK_PATH
    else:
        raise ValueError(f"Unsupported coursework dataset '{dataset}'.")

    df = pd.read_csv(path, encoding="latin-1")
    return _prepare_coursework_frame(df)


def split_allowed_courses(value: str) -> set[str]:
    return {normalize_course_number(item) for item in str(value).split("|") if item}


def build_course_lookup() -> dict[str, dict]:
    catalog = load_catalog()
    return catalog.set_index("course_number").to_dict(orient="index")


def term_sort_key(term: str) -> tuple[int, int]:
    year_text, season = str(term).split("-", maxsplit=1)
    season_order = {"SPRING": 1, "SUMMER": 2, "FALL": 3}
    return int(year_text), season_order.get(season.upper(), 9)


def transcript_gpa(transcript_df: pd.DataFrame) -> float | None:
    graded = transcript_df.dropna(subset=["grade_points"])
    if graded.empty:
        return None
    return round(float(graded["grade_points"].mean()), 2)
