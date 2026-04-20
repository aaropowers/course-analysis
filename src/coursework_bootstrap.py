from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    BOOTSTRAPPED_COURSEWORK_PATH,
    CLEAN_COURSEWORK_PATH,
    GRADE_POINTS,
    RAW_COURSEWORK_PATH,
    build_course_lookup,
    derive_department,
    load_catalog,
    load_degree_plan,
    load_prereqs,
    normalize_course_number,
    normalize_grade,
    normalize_semester,
)


TRACKED_OUTSIDE_CREDIT_COURSES = {
    "M408C",
    "M408D",
    "CH301",
    "PHY303K",
    "PHY103M",
    "PHY303L",
    "PHY103N",
}
CRITICAL_COLUMNS = ["student_id", "year", "semester", "course_number"]
LETTER_GRADE_ORDER = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F", "CR", "Q", "W", "IP"]
TARGET_TOTAL_ROWS = 28000
RNG_SEED = 42
TERM_SEQUENCE = ["SPRING", "SUMMER", "FALL"]
GRADE_POINTS_ONLY = {grade: points for grade, points in GRADE_POINTS.items() if points is not None}


@dataclass(frozen=True)
class BootstrapSummary:
    cleaned_rows: int
    cleaned_students: int
    bootstrapped_rows: int
    bootstrapped_students: int
    synthetic_students_added: int
    template_students_used: int


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split()).strip()


def _validate_catalog_coverage() -> None:
    catalog_courses = set(load_catalog()["course_number"])
    degree_courses = {
        normalize_course_number(course)
        for value in load_degree_plan()["allowed_courses"]
        for course in str(value).split("|")
    }
    missing = sorted(course for course in degree_courses if course and course not in catalog_courses)
    if missing:
        raise ValueError(
            "The degree plan references courses missing from data/course_catalog.csv: " + ", ".join(missing)
        )


def _build_fallback_course_metadata(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby("course_number")
    rows = []
    for course_number, frame in grouped:
        desc_counts = frame["course_description"].map(_clean_text)
        desc_counts = desc_counts[desc_counts != ""].value_counts()
        description = ""
        if not desc_counts.empty:
            top_count = desc_counts.iloc[0]
            candidates = sorted([desc for desc, count in desc_counts.items() if count == top_count], key=len, reverse=True)
            description = candidates[0]
        credits_mode = frame["credits"].dropna().mode()
        rows.append(
            {
                "course_number": course_number,
                "course_title": course_number,
                "description": description,
                "credits": float(credits_mode.iloc[0]) if not credits_mode.empty else pd.NA,
                "department": derive_department(course_number),
            }
        )
    return pd.DataFrame(rows)


def _build_suffix_course_map() -> dict[str, str]:
    suffix_map: defaultdict[str, set[str]] = defaultdict(set)
    for course_number in load_catalog()["course_number"]:
        suffix = "".join(char for char in str(course_number) if char.isdigit() or char.isalpha())  # normalized already
        numeric_suffix = suffix.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if numeric_suffix:
            suffix_map[numeric_suffix].add(course_number)
    return {
        suffix: next(iter(courses))
        for suffix, courses in suffix_map.items()
        if len(courses) == 1
    }


def clean_coursework_dataset(raw_path: Path = RAW_COURSEWORK_PATH) -> pd.DataFrame:
    _validate_catalog_coverage()

    raw_df = pd.read_csv(raw_path, encoding="latin-1")
    raw_df.columns = [column.strip().lower().replace(" ", "_") for column in raw_df.columns]

    for column in ["institution", "credits", "grade", "course_description"]:
        if column not in raw_df.columns:
            raw_df[column] = pd.NA

    suffix_map = _build_suffix_course_map()
    raw_df["course_number"] = raw_df["course_number"].map(normalize_course_number)
    raw_df["course_number"] = raw_df["course_number"].map(lambda value: suffix_map.get(value, value))
    raw_df["semester"] = raw_df["semester"].map(normalize_semester)
    raw_df["grade"] = raw_df["grade"].map(normalize_grade)
    raw_df["institution"] = raw_df["institution"].fillna("utexas").astype(str).str.strip().str.lower().replace({"": "utexas"})
    raw_df["year"] = pd.to_numeric(raw_df["year"], errors="coerce").astype("Int64")
    raw_df["credits"] = pd.to_numeric(raw_df["credits"], errors="coerce")
    raw_df["course_description"] = raw_df["course_description"].map(_clean_text)

    cleaned_df = raw_df.dropna(subset=CRITICAL_COLUMNS).copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    catalog_df = load_catalog().rename(columns={"description": "catalog_description"})
    fallback_df = _build_fallback_course_metadata(cleaned_df).rename(
        columns={
            "course_title": "fallback_course_title",
            "description": "fallback_description",
            "credits": "fallback_credits",
            "department": "fallback_department",
        }
    )

    cleaned_df = cleaned_df.merge(
        catalog_df[["course_number", "course_title", "catalog_description", "credits", "department"]],
        on="course_number",
        how="left",
        suffixes=("", "_catalog"),
    )
    cleaned_df = cleaned_df.merge(fallback_df, on="course_number", how="left")

    cleaned_df["course_title"] = cleaned_df["course_title"].fillna(cleaned_df["fallback_course_title"]).fillna(cleaned_df["course_number"])
    cleaned_df["course_description"] = cleaned_df["catalog_description"].fillna(cleaned_df["fallback_description"]).fillna("")
    cleaned_df["department"] = cleaned_df["department"].fillna(cleaned_df["fallback_department"]).fillna(cleaned_df["course_number"].map(derive_department))
    cleaned_df["credits"] = cleaned_df["credits"].fillna(cleaned_df["fallback_credits"])
    cleaned_df["record_source"] = "raw_cleaned"

    cleaned_df = cleaned_df[
        [
            "student_id",
            "year",
            "semester",
            "institution",
            "course_number",
            "credits",
            "grade",
            "course_description",
            "department",
            "course_title",
            "record_source",
        ]
    ].sort_values(["student_id", "year", "semester", "course_number"], kind="stable")
    return cleaned_df.reset_index(drop=True)


def _term_order_key(frame: pd.DataFrame) -> pd.DataFrame:
    season_order = {"SPRING": 1, "SUMMER": 2, "FALL": 3}
    ordered = frame.copy()
    ordered["semester_order"] = ordered["semester"].map(season_order).fillna(9)
    sort_cols = ["year", "semester_order"]
    if "course_number" in ordered.columns:
        sort_cols.append("course_number")
    return ordered.sort_values(sort_cols, kind="stable")


def _build_grade_sampler(cleaned_df: pd.DataFrame) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, tuple[list[str], np.ndarray]], tuple[list[str], np.ndarray]]:
    valid_grades = cleaned_df["grade"].dropna()
    valid_grades = valid_grades[valid_grades.isin(LETTER_GRADE_ORDER)]

    def to_distribution(series: pd.Series) -> tuple[list[str], np.ndarray]:
        counts = series.value_counts().reindex(LETTER_GRADE_ORDER, fill_value=0)
        counts = counts[counts > 0]
        labels = counts.index.tolist()
        probs = (counts / counts.sum()).to_numpy(dtype=float)
        return labels, probs

    by_course: dict[str, tuple[list[str], np.ndarray]] = {}
    for course_number, frame in cleaned_df.groupby("course_number"):
        grades = frame["grade"].dropna()
        grades = grades[grades.isin(LETTER_GRADE_ORDER)]
        if not grades.empty:
            by_course[course_number] = to_distribution(grades)

    by_department: dict[str, tuple[list[str], np.ndarray]] = {}
    for department, frame in cleaned_df.groupby("department"):
        grades = frame["grade"].dropna()
        grades = grades[grades.isin(LETTER_GRADE_ORDER)]
        if not grades.empty:
            by_department[department] = to_distribution(grades)

    global_dist = to_distribution(valid_grades)
    return by_course, by_department, global_dist


def _estimate_course_target_gpa(
    course_number: str,
    catalog_lookup: dict[str, dict],
    prereq_map: dict[str, dict[str, set[str]]],
    observed_course_gpa: dict[str, float],
    observed_department_gpa: dict[str, float],
    global_gpa: float,
) -> float:
    catalog_meta = catalog_lookup.get(course_number, {})
    catalog_avg = pd.to_numeric(pd.Series([catalog_meta.get("avg_gpa")]), errors="coerce").iloc[0]
    dept = catalog_meta.get("department") or derive_department(course_number)
    dept_gpa = observed_department_gpa.get(dept, global_gpa)

    prereq_courses = sorted(prereq_map.get(course_number, {}).get("PREREQ", set()) | prereq_map.get(course_number, {}).get("COREQ", set()))
    prereq_gpas = [observed_course_gpa[prereq] for prereq in prereq_courses if prereq in observed_course_gpa]
    prereq_gpa = float(np.mean(prereq_gpas)) if prereq_gpas else dept_gpa

    components = [float(value) for value in [catalog_avg, prereq_gpa, dept_gpa] if pd.notna(value)]
    target_gpa = float(np.mean(components)) if components else global_gpa

    description = str(catalog_meta.get("description", "")).lower()
    harder_keywords = {
        "thermo",
        "fluid",
        "heat transfer",
        "dynamics",
        "control",
        "statistics",
        "nuclear",
        "analysis",
        "computational",
    }
    easier_keywords = {"design", "laboratory", "communication", "policy", "seminar", "project", "graphics"}
    if any(keyword in description for keyword in harder_keywords):
        target_gpa -= 0.12
    if any(keyword in description for keyword in easier_keywords):
        target_gpa += 0.08

    recommended_semester = pd.to_numeric(pd.Series([catalog_meta.get("recommended_semester")]), errors="coerce").iloc[0]
    if pd.notna(recommended_semester) and float(recommended_semester) >= 7:
        target_gpa -= 0.05

    if course_number.endswith("L"):
        target_gpa += 0.1

    return float(np.clip(target_gpa, 2.0, 3.85))


def _synthesize_grade_distribution(
    target_gpa: float,
    department_distribution: tuple[list[str], np.ndarray] | None,
    global_distribution: tuple[list[str], np.ndarray],
) -> tuple[list[str], np.ndarray]:
    numeric_grades = list(GRADE_POINTS_ONLY.keys())
    grade_values = np.array([GRADE_POINTS_ONLY[grade] for grade in numeric_grades], dtype=float)
    sigma = 0.52 if target_gpa >= 3.2 else 0.6
    gaussian_weights = np.exp(-((grade_values - target_gpa) ** 2) / (2 * sigma**2))
    gaussian_probs = gaussian_weights / gaussian_weights.sum()

    base_labels, base_probs = department_distribution or global_distribution
    base_map = {label: prob for label, prob in zip(base_labels, base_probs)}
    blended = np.array(
        [0.55 * gaussian_probs[index] + 0.45 * base_map.get(grade, 0.0) for index, grade in enumerate(numeric_grades)],
        dtype=float,
    )
    blended = blended / blended.sum()
    return numeric_grades, blended


def _add_synthetic_grade_distributions(
    cleaned_df: pd.DataFrame,
    grade_dists: tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, tuple[list[str], np.ndarray]], tuple[list[str], np.ndarray]],
) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, tuple[list[str], np.ndarray]], tuple[list[str], np.ndarray], list[str]]:
    by_course, by_department, global_dist = grade_dists
    catalog_lookup = build_course_lookup()
    prereq_map = _normalize_prereq_table(load_prereqs())

    graded = cleaned_df[cleaned_df["grade"].map(lambda grade: grade in GRADE_POINTS_ONLY)].copy()
    observed_course_gpa = graded.groupby("course_number")["grade"].apply(
        lambda series: float(np.mean([GRADE_POINTS_ONLY[grade] for grade in series if grade in GRADE_POINTS_ONLY]))
    ).to_dict()
    observed_department_gpa = graded.groupby("department")["grade"].apply(
        lambda series: float(np.mean([GRADE_POINTS_ONLY[grade] for grade in series if grade in GRADE_POINTS_ONLY]))
    ).to_dict()
    global_gpa = float(np.mean([GRADE_POINTS_ONLY[grade] for grade in graded["grade"] if grade in GRADE_POINTS_ONLY])) if not graded.empty else 3.0

    degree_courses = sorted(
        {
            normalize_course_number(course)
            for value in load_degree_plan()["allowed_courses"]
            for course in str(value).split("|")
            if normalize_course_number(course)
        }
    )
    synthesized_courses: list[str] = []
    for course_number in degree_courses:
        if course_number in by_course:
            continue
        meta = catalog_lookup.get(course_number)
        if not meta:
            continue
        department = meta.get("department") or derive_department(course_number)
        target_gpa = _estimate_course_target_gpa(
            course_number,
            catalog_lookup,
            prereq_map,
            observed_course_gpa,
            observed_department_gpa,
            global_gpa,
        )
        by_course[course_number] = _synthesize_grade_distribution(target_gpa, by_department.get(department), global_dist)
        synthesized_courses.append(course_number)

    return by_course, by_department, global_dist, synthesized_courses


def _sample_grade(course_number: str, department: str, grade_dists: tuple[dict, dict, tuple], rng: np.random.Generator) -> str | None:
    course_dist, dept_dist, global_dist = grade_dists
    labels, probs = course_dist.get(course_number) or dept_dist.get(department) or global_dist
    return str(rng.choice(labels, p=probs)) if labels else None


def _normalize_prereq_table(prereq_df: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    table: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"PREREQ": set(), "COREQ": set()})
    for row in prereq_df.itertuples():
        table[row.course_number][row.prereq_type].add(row.prerequisite_course)
    return table


def _trackable_courses() -> set[str]:
    prereq_df = load_prereqs()
    tracked = set(prereq_df["course_number"]).union(set(prereq_df["prerequisite_course"]))
    tracked.update({"ME333T", "ME366J", "ME266K", "ME266P", "UGS303", "RHE306", "GOV310L", "HIS315L"})
    return {course for course in tracked if course}


def _student_term_sequence(student_df: pd.DataFrame) -> list[tuple[int, str]]:
    ordered = _term_order_key(student_df[["year", "semester"]].drop_duplicates())
    return [(int(row.year), str(row.semester)) for row in ordered.itertuples()]


def _student_has_valid_tracked_order(student_df: pd.DataFrame, prereq_map: dict[str, dict[str, set[str]]]) -> bool:
    tracked = student_df[student_df["course_number"].isin(_trackable_courses())].copy()
    if tracked.empty:
        return False

    ordered_terms = _student_term_sequence(tracked)
    term_index = {term: index for index, term in enumerate(ordered_terms)}
    completed_or_waived = set(TRACKED_OUTSIDE_CREDIT_COURSES)
    term_courses: list[set[str]] = []
    for year, semester in ordered_terms:
        courses = set(tracked[(tracked["year"] == year) & (tracked["semester"] == semester)]["course_number"])
        term_courses.append(courses)

    for current_index, courses in enumerate(term_courses):
        current_term = courses
        for course in current_term:
            prereqs = prereq_map.get(course, {}).get("PREREQ", set())
            if any(prereq not in completed_or_waived for prereq in prereqs):
                return False
            coreqs = prereq_map.get(course, {}).get("COREQ", set())
            if any(coreq not in completed_or_waived and coreq not in current_term for coreq in coreqs):
                return False
        completed_or_waived.update(current_term)
    return True


def _build_template_pool(cleaned_df: pd.DataFrame) -> list[pd.DataFrame]:
    prereq_map = _normalize_prereq_table(load_prereqs())
    templates: list[pd.DataFrame] = []
    for _, frame in cleaned_df.groupby("student_id"):
        student_df = _term_order_key(frame)
        if _student_has_valid_tracked_order(student_df, prereq_map):
            templates.append(student_df.reset_index(drop=True))
    if not templates:
        raise ValueError("No valid student templates were found after cleaning the raw coursework data.")
    return templates


def _sample_start_years(cleaned_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    first_terms = (
        _term_order_key(cleaned_df[["student_id", "year", "semester"]].drop_duplicates())
        .groupby("student_id")
        .first()
        .reset_index()
    )
    counts = first_terms["year"].value_counts().sort_index()
    return counts.index.to_numpy(dtype=int), (counts / counts.sum()).to_numpy(dtype=float)


def _shift_term_sequence(template_terms: list[tuple[int, str]], start_year: int) -> dict[tuple[int, str], tuple[int, str]]:
    base_year = template_terms[0][0]
    year_shift = int(start_year) - int(base_year)
    return {(year, semester): (int(year) + year_shift, semester) for year, semester in template_terms}


def _next_term(year: int, semester: str) -> tuple[int, str]:
    index = TERM_SEQUENCE.index(semester)
    if index == len(TERM_SEQUENCE) - 1:
        return year + 1, TERM_SEQUENCE[0]
    return year, TERM_SEQUENCE[index + 1]


def _term_code(year: int, semester: str) -> int:
    return int(year) * 10 + TERM_SEQUENCE.index(semester) + 1


def _ensure_term_exists(student_df: pd.DataFrame, year: int, semester: str) -> pd.DataFrame:
    existing = student_df[(student_df["year"] == year) & (student_df["semester"] == semester)]
    if not existing.empty:
        return student_df
    placeholder = pd.DataFrame(
        [{"student_id": student_df["student_id"].iloc[0], "year": year, "semester": semester, "course_number": pd.NA}]
    )
    return pd.concat([student_df, placeholder], ignore_index=True)


def _find_eligible_term_for_course(student_df: pd.DataFrame, course_number: str, prereq_map: dict[str, dict[str, set[str]]]) -> tuple[int, str] | None:
    student_df = student_df.copy()
    student_df = student_df.dropna(subset=["course_number"])
    if course_number in set(student_df["course_number"]):
        return None

    terms = _student_term_sequence(student_df)
    if not terms:
        return None

    course_first_term: dict[str, int] = {}
    for row in student_df.itertuples():
        code = _term_code(int(row.year), str(row.semester))
        course_first_term[row.course_number] = min(code, course_first_term.get(row.course_number, code))

    prereqs = prereq_map.get(course_number, {}).get("PREREQ", set())
    coreqs = prereq_map.get(course_number, {}).get("COREQ", set())

    candidate_terms = list(terms)
    latest_year, latest_semester = candidate_terms[-1]
    candidate_terms.append(_next_term(latest_year, latest_semester))

    for year, semester in candidate_terms:
        current_code = _term_code(year, semester)
        if any(course_first_term.get(prereq, -1) >= current_code or prereq not in course_first_term for prereq in prereqs):
            continue
        coreq_valid = True
        for coreq in coreqs:
            coreq_term = course_first_term.get(coreq)
            if coreq_term is None:
                coreq_valid = False
                break
            if coreq_term > current_code:
                coreq_valid = False
                break
        if coreq_valid:
            return year, semester
    return None


def _recommended_semester_lookup() -> dict[str, int]:
    lookup: dict[str, int] = {}
    catalog = load_catalog()
    if "recommended_semester" in catalog.columns:
        for row in catalog.itertuples():
            if pd.notna(row.recommended_semester):
                lookup[str(row.course_number)] = int(row.recommended_semester)
    return lookup


def _augment_missing_degree_courses(
    combined_df: pd.DataFrame,
    grade_dists: tuple[dict, dict, tuple],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[str]]:
    degree_courses = sorted(
        {
            normalize_course_number(course)
            for value in load_degree_plan()["allowed_courses"]
            for course in str(value).split("|")
            if normalize_course_number(course)
        }
    )
    current_courses = set(combined_df["course_number"].dropna())
    missing_courses = [course for course in degree_courses if course not in current_courses]
    if not missing_courses:
        return combined_df, []

    prereq_map = _normalize_prereq_table(load_prereqs())
    catalog_lookup = build_course_lookup()
    recommended_semesters = _recommended_semester_lookup()
    synthetic_students = [student_id for student_id in combined_df["student_id"].unique() if str(student_id).startswith("boot_")]
    if not synthetic_students:
        return combined_df, []

    augmented_rows: list[dict] = []
    for course_number in missing_courses:
        meta = catalog_lookup.get(course_number, {})
        department = meta.get("department") or derive_department(course_number)
        recommended_semester = recommended_semesters.get(course_number, 7)

        analogous = combined_df[
            (combined_df["department"] == department)
            & (combined_df["course_number"] != course_number)
        ].copy()
        analogous["recommended_semester"] = analogous["course_number"].map(lambda value: recommended_semesters.get(value))
        analogous = analogous[analogous["recommended_semester"].notna()]
        analogous = analogous[(analogous["recommended_semester"] - recommended_semester).abs() <= 1]
        analogous_counts = analogous.groupby("course_number").size()
        target_count = int(np.clip(analogous_counts.median() if not analogous_counts.empty else 90, 80, 220))

        candidate_assignments: list[tuple[str, tuple[int, str]]] = []
        for student_id in synthetic_students:
            student_df = combined_df[combined_df["student_id"] == student_id][["student_id", "year", "semester", "course_number"]].copy()
            term = _find_eligible_term_for_course(student_df, course_number, prereq_map)
            if term is None:
                continue
            candidate_assignments.append((student_id, term))

        if not candidate_assignments:
            continue

        rng.shuffle(candidate_assignments)
        selected_assignments = candidate_assignments[: min(target_count, len(candidate_assignments))]
        for student_id, (year, semester) in selected_assignments:
            grade = _sample_grade(course_number, department, grade_dists, rng)
            augmented_rows.append(
                {
                    "student_id": student_id,
                    "year": int(year),
                    "semester": semester,
                    "institution": "utexas",
                    "course_number": course_number,
                    "credits": meta.get("credits", 3),
                    "grade": grade,
                    "course_description": meta.get("description", ""),
                    "department": department,
                    "course_title": meta.get("course_title", course_number),
                    "record_source": "synthetic_bootstrap",
                }
            )

    if not augmented_rows:
        return combined_df, []

    augmented_df = pd.concat([combined_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    augmented_df = augmented_df.sort_values(["student_id", "year", "semester", "course_number"], kind="stable").reset_index(drop=True)
    added_courses = sorted(set(row["course_number"] for row in augmented_rows))
    return augmented_df, added_courses


def bootstrap_coursework_dataset(
    cleaned_df: pd.DataFrame,
    target_total_rows: int = TARGET_TOTAL_ROWS,
    seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, BootstrapSummary]:
    rng = np.random.default_rng(seed)
    raw_grade_dists = _build_grade_sampler(cleaned_df)
    course_grade_dists, dept_grade_dists, global_grade_dist, _ = _add_synthetic_grade_distributions(cleaned_df, raw_grade_dists)
    grade_dists = (course_grade_dists, dept_grade_dists, global_grade_dist)
    templates = _build_template_pool(cleaned_df)
    year_values, year_probs = _sample_start_years(cleaned_df)
    avg_rows_per_student = float(cleaned_df.groupby("student_id").size().mean())
    synthetic_student_count = max(1, int(np.ceil((target_total_rows - len(cleaned_df)) / avg_rows_per_student)))
    course_lookup = build_course_lookup()

    synthetic_rows: list[dict] = []
    template_use_counter: Counter[int] = Counter()
    for index in range(synthetic_student_count):
        template_index = int(rng.integers(0, len(templates)))
        template_use_counter[template_index] += 1
        template = templates[template_index].copy()
        template_terms = _student_term_sequence(template)
        start_year = int(rng.choice(year_values, p=year_probs))
        term_map = _shift_term_sequence(template_terms, start_year)
        synthetic_student_id = f"boot_{start_year}_{index:04d}"

        for row in template.itertuples():
            mapped_year, mapped_semester = term_map[(int(row.year), str(row.semester))]
            course_number = str(row.course_number)
            catalog_meta = course_lookup.get(course_number, {})
            department = catalog_meta.get("department") or derive_department(course_number)
            grade = _sample_grade(course_number, department, grade_dists, rng)
            synthetic_rows.append(
                {
                    "student_id": synthetic_student_id,
                    "year": mapped_year,
                    "semester": mapped_semester,
                    "institution": "utexas",
                    "course_number": course_number,
                    "credits": row.credits,
                    "grade": grade,
                    "course_description": catalog_meta.get("description", row.course_description),
                    "department": department,
                    "course_title": catalog_meta.get("course_title", row.course_title),
                    "record_source": "synthetic_bootstrap",
                }
            )

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined_df = pd.concat([cleaned_df, synthetic_df], ignore_index=True)
    combined_df, _ = _augment_missing_degree_courses(combined_df, grade_dists, rng)
    combined_df["year"] = combined_df["year"].astype(int)
    combined_df["credits"] = pd.to_numeric(combined_df["credits"], errors="coerce")
    combined_df = combined_df.sort_values(["student_id", "year", "semester", "course_number"], kind="stable").reset_index(drop=True)

    summary = BootstrapSummary(
        cleaned_rows=int(len(cleaned_df)),
        cleaned_students=int(cleaned_df["student_id"].nunique()),
        bootstrapped_rows=int(len(combined_df)),
        bootstrapped_students=int(combined_df["student_id"].nunique()),
        synthetic_students_added=int(synthetic_student_count),
        template_students_used=int(sum(1 for count in template_use_counter.values() if count > 0)),
    )
    return combined_df, summary


def validate_bootstrapped_dataset(cleaned_df: pd.DataFrame, bootstrapped_df: pd.DataFrame) -> dict[str, object]:
    catalog_df = load_catalog()
    catalog_map = catalog_df.set_index("course_number")["description"].to_dict()
    bad_catalog_descriptions = []
    for course_number, description in catalog_map.items():
        course_rows = bootstrapped_df[bootstrapped_df["course_number"] == course_number]
        if course_rows.empty:
            continue
        if course_rows["course_description"].nunique(dropna=False) != 1 or course_rows["course_description"].iloc[0] != description:
            bad_catalog_descriptions.append(course_number)

    cleaned_grade_values = sorted(set(grade for grade in cleaned_df["grade"].dropna().tolist() if grade not in set(GRADE_POINTS) | {"Q", "W", "IP"}))
    semester_values = sorted(set(bootstrapped_df["semester"].dropna()))
    prereq_map = _normalize_prereq_table(load_prereqs())
    invalid_students = []
    for student_id, frame in bootstrapped_df.groupby("student_id"):
        if str(student_id).startswith("boot_") and not _student_has_valid_tracked_order(frame, prereq_map):
            invalid_students.append(student_id)

    return {
        "invalid_cleaned_grades": cleaned_grade_values,
        "invalid_semesters": [semester for semester in semester_values if semester not in {"FALL", "SPRING", "SUMMER"}],
        "catalog_description_mismatches": bad_catalog_descriptions,
        "multi_description_courses": bootstrapped_df.groupby("course_number")["course_description"].nunique().loc[lambda s: s > 1].index.tolist(),
        "missing_departments": int((bootstrapped_df["department"].fillna("") == "").sum()),
        "invalid_bootstrap_students": invalid_students[:20],
    }


def write_coursework_outputs(
    cleaned_df: pd.DataFrame,
    bootstrapped_df: pd.DataFrame,
    clean_path: Path = CLEAN_COURSEWORK_PATH,
    bootstrapped_path: Path = BOOTSTRAPPED_COURSEWORK_PATH,
) -> None:
    cleaned_df.to_csv(clean_path, index=False, encoding="latin-1")
    bootstrapped_df.to_csv(bootstrapped_path, index=False, encoding="latin-1")


def generate_coursework_assets() -> tuple[BootstrapSummary, dict[str, object]]:
    cleaned_df = clean_coursework_dataset()
    bootstrapped_df, summary = bootstrap_coursework_dataset(cleaned_df)
    write_coursework_outputs(cleaned_df, bootstrapped_df)
    validation = validate_bootstrapped_dataset(cleaned_df, bootstrapped_df)
    return summary, validation
