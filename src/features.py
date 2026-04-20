from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_catalog, load_transcripts, normalize_course_number

GRADE_ORDER = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]
SEMESTER_ORDER = {"SPRING": 1, "SUMMER": 2, "FALL": 3}


def compute_course_similarity(catalog_df: pd.DataFrame | None = None) -> pd.DataFrame:
    catalog_df = catalog_df if catalog_df is not None else load_catalog()
    descriptions = catalog_df["description"].fillna("")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(descriptions)
    similarity = cosine_similarity(tfidf)
    return pd.DataFrame(similarity, index=catalog_df["course_number"], columns=catalog_df["course_number"])


def build_coenrollment_features(transcript_df: pd.DataFrame | None = None) -> pd.DataFrame:
    transcript_df = transcript_df if transcript_df is not None else load_transcripts()
    pair_counts: Counter[tuple[str, str]] = Counter()

    group_cols = ["student_id", "term"] if "term" in transcript_df.columns else ["student_id", "year", "semester"]
    for _, frame in transcript_df.groupby(group_cols):
        unique_courses = sorted(set(frame["course_number"]))
        for course_a, course_b in combinations(unique_courses, 2):
            pair_counts[(course_a, course_b)] += 1

    rows = [
        {"course_a": course_a, "course_b": course_b, "coenrollment_count": count}
        for (course_a, course_b), count in pair_counts.items()
    ]
    return pd.DataFrame(rows).sort_values("coenrollment_count", ascending=False).reset_index(drop=True)


def build_student_course_matrix(transcript_df: pd.DataFrame | None = None) -> pd.DataFrame:
    transcript_df = transcript_df if transcript_df is not None else load_transcripts()
    graded = transcript_df.dropna(subset=["grade_points"])
    if graded.empty:
        return pd.DataFrame()
    return graded.pivot_table(index="student_id", columns="course_number", values="grade_points", aggfunc="mean")


def interest_overlap(course_interest_tags: str, selected_interests: list[str]) -> int:
    if not selected_interests:
        return 0
    course_tags = {normalize_course_number(tag) for tag in str(course_interest_tags).split("|")}
    desired = {normalize_course_number(tag) for tag in selected_interests}
    return len(course_tags & desired)


def filter_letter_grade_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["grade"].isin(GRADE_ORDER)].copy()


def build_course_grade_distribution(coursework_df: pd.DataFrame, course_number: str) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    course_number = normalize_course_number(course_number)
    graded = filter_letter_grade_records(coursework_df)
    graded = graded[graded["course_number"] == course_number].copy()
    if graded.empty:
        return pd.DataFrame(columns=["grade", "count", "cumulative_pct"]), {"sample_size": 0, "avg_gpa": None, "pass_rate": None}

    counts = graded["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    counts = counts[counts > 0]
    distribution = pd.DataFrame({"grade": counts.index, "count": counts.values})
    distribution["cumulative_pct"] = distribution["count"].cumsum() / distribution["count"].sum() * 100

    avg_gpa = round(float(graded["grade_points"].mean()), 2) if graded["grade_points"].notna().any() else None
    pass_rate = round(float((graded["grade_points"] >= 1.7).mean() * 100), 1) if not graded.empty else None
    summary = {
        "sample_size": int(len(graded)),
        "avg_gpa": avg_gpa,
        "pass_rate": pass_rate,
    }
    return distribution, summary


def build_student_semester_bundles(coursework_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        coursework_df.groupby(["student_id", "year", "semester"])["course_number"]
        .apply(lambda values: sorted(set(values)))
        .reset_index()
        .rename(columns={"course_number": "courses_taken_together"})
    )
    return grouped


def build_coenrollment_effect(coursework_df: pd.DataFrame, target_course: str, companion_course: str) -> dict[str, object]:
    target_course = normalize_course_number(target_course)
    companion_course = normalize_course_number(companion_course)
    graded = filter_letter_grade_records(coursework_df)
    target_df = graded[graded["course_number"] == target_course].copy()
    if target_df.empty:
        return {
            "comparison_df": pd.DataFrame(columns=["grade", "group", "count"]),
            "summary": {"target_course": target_course, "companion_course": companion_course, "with_count": 0, "without_count": 0, "with_avg_gpa": None, "without_avg_gpa": None, "gpa_delta": None},
        }

    bundles = build_student_semester_bundles(coursework_df)
    with_companion = set()
    for row in bundles.itertuples():
        courses = set(row.courses_taken_together)
        if target_course in courses and companion_course in courses:
            with_companion.add((row.student_id, row.year, row.semester))

    target_df["group"] = target_df.apply(
        lambda row: "With companion" if (row["student_id"], row["year"], row["semester"]) in with_companion else "Without companion",
        axis=1,
    )

    comparison_rows = []
    for group_name, frame in target_df.groupby("group"):
        counts = frame["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
        counts = counts[counts > 0]
        comparison_rows.extend(
            {"grade": grade, "group": group_name, "count": int(count)}
            for grade, count in counts.items()
        )

    with_group = target_df[target_df["group"] == "With companion"]
    without_group = target_df[target_df["group"] == "Without companion"]
    with_avg = round(float(with_group["grade_points"].mean()), 2) if not with_group.empty else None
    without_avg = round(float(without_group["grade_points"].mean()), 2) if not without_group.empty else None

    return {
        "comparison_df": pd.DataFrame(comparison_rows),
        "summary": {
            "target_course": target_course,
            "companion_course": companion_course,
            "with_count": int(len(with_group)),
            "without_count": int(len(without_group)),
            "with_avg_gpa": with_avg,
            "without_avg_gpa": without_avg,
            "gpa_delta": round(with_avg - without_avg, 2) if with_avg is not None and without_avg is not None else None,
        },
    }


def build_department_gpa_heatmap(coursework_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    graded = filter_letter_grade_records(coursework_df)
    if graded.empty:
        return pd.DataFrame()

    dept_counts = graded.groupby("department")["grade_points"].count().sort_values(ascending=False)
    top_departments = dept_counts.head(top_n).index.tolist()
    heatmap = (
        graded[graded["department"].isin(top_departments)]
        .groupby(["department", "semester"])["grade_points"]
        .mean()
        .reset_index()
        .pivot(index="department", columns="semester", values="grade_points")
    )
    ordered_columns = [semester for semester in ["SPRING", "SUMMER", "FALL"] if semester in heatmap.columns]
    heatmap = heatmap.reindex(columns=ordered_columns)
    heatmap = heatmap.loc[heatmap.mean(axis=1).sort_values(ascending=False).index]
    return heatmap


def get_top_courses_by_enrollment(coursework_df: pd.DataFrame, n: int = 25, graded_only: bool = False) -> list[str]:
    frame = filter_letter_grade_records(coursework_df) if graded_only else coursework_df
    return frame["course_number"].value_counts().head(n).index.tolist()


def _term_code(frame: pd.DataFrame) -> pd.Series:
    season_code = frame["semester"].map(SEMESTER_ORDER).fillna(9)
    return frame["year"].fillna(0).astype(int) * 10 + season_code.astype(int)


def build_same_term_companion_counts(coursework_df: pd.DataFrame, course_number: str) -> pd.DataFrame:
    course_number = normalize_course_number(course_number)
    bundles = build_student_semester_bundles(coursework_df)
    companion_counter: Counter[str] = Counter()

    for row in bundles.itertuples():
        courses = set(row.courses_taken_together)
        if course_number not in courses:
            continue
        for companion in courses:
            if companion != course_number:
                companion_counter[companion] += 1

    if not companion_counter:
        return pd.DataFrame(columns=["course_number", "count"])

    return (
        pd.DataFrame(
            [{"course_number": course, "count": count} for course, count in companion_counter.items()]
        )
        .sort_values(["count", "course_number"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_anchor_sequence_support(coursework_df: pd.DataFrame, anchor_courses: list[str], target_course: str) -> dict[str, object]:
    target_course = normalize_course_number(target_course)
    normalized_anchors = [normalize_course_number(course) for course in anchor_courses if normalize_course_number(course)]
    if not normalized_anchors:
        return {
            "anchor_course": None,
            "after_anchor_count": 0,
            "anchor_target_students": 0,
            "share_after_anchor": None,
        }

    course_terms = coursework_df[["student_id", "course_number", "year", "semester"]].copy()
    course_terms["term_code"] = _term_code(course_terms)
    course_terms = (
        course_terms.groupby(["student_id", "course_number"], as_index=False)["term_code"]
        .min()
    )

    target_terms = course_terms[course_terms["course_number"] == target_course][["student_id", "term_code"]].rename(
        columns={"term_code": "target_term_code"}
    )
    if target_terms.empty:
        return {
            "anchor_course": None,
            "after_anchor_count": 0,
            "anchor_target_students": 0,
            "share_after_anchor": None,
        }

    best_support = {
        "anchor_course": None,
        "after_anchor_count": 0,
        "anchor_target_students": 0,
        "share_after_anchor": None,
    }

    for anchor in normalized_anchors:
        anchor_terms = course_terms[course_terms["course_number"] == anchor][["student_id", "term_code"]].rename(
            columns={"term_code": "anchor_term_code"}
        )
        if anchor_terms.empty:
            continue

        joined = target_terms.merge(anchor_terms, on="student_id", how="inner")
        if joined.empty:
            continue

        after_count = int((joined["target_term_code"] > joined["anchor_term_code"]).sum())
        total_pairs = int(len(joined))
        share = round(after_count / total_pairs * 100, 1) if total_pairs else None

        if after_count > int(best_support["after_anchor_count"]):
            best_support = {
                "anchor_course": anchor,
                "after_anchor_count": after_count,
                "anchor_target_students": total_pairs,
                "share_after_anchor": share,
            }

    return best_support


def _difficulty_label(avg_gpa: float | None, pass_rate: float | None) -> str | None:
    if avg_gpa is None and pass_rate is None:
        return None
    if avg_gpa is not None and avg_gpa >= 3.3 and (pass_rate is None or pass_rate >= 92):
        return "Supportive grading"
    if avg_gpa is not None and avg_gpa <= 2.7:
        return "Higher-challenge"
    if pass_rate is not None and pass_rate < 85:
        return "Higher-challenge"
    return "Moderate load"


def summarize_course_evidence(coursework_df: pd.DataFrame, course_number: str, anchor_courses: list[str] | None = None) -> dict[str, object]:
    course_number = normalize_course_number(course_number)
    _, grade_summary = build_course_grade_distribution(coursework_df, course_number)
    companions = build_same_term_companion_counts(coursework_df, course_number)
    anchor_support = build_anchor_sequence_support(coursework_df, anchor_courses or [], course_number)

    top_companion = None
    top_companion_count = 0
    if not companions.empty:
        top_companion = companions.iloc[0]["course_number"]
        top_companion_count = int(companions.iloc[0]["count"])

    avg_gpa = grade_summary["avg_gpa"]
    pass_rate = grade_summary["pass_rate"]
    return {
        "evidence_avg_gpa": avg_gpa,
        "evidence_pass_rate": pass_rate,
        "evidence_records": int(grade_summary["sample_size"]),
        "evidence_top_companion": top_companion,
        "evidence_top_companion_count": top_companion_count,
        "evidence_anchor_course": anchor_support["anchor_course"],
        "evidence_after_anchor_count": int(anchor_support["after_anchor_count"]),
        "evidence_anchor_target_students": int(anchor_support["anchor_target_students"]),
        "evidence_share_after_anchor": anchor_support["share_after_anchor"],
        "evidence_difficulty": _difficulty_label(avg_gpa, pass_rate),
    }


def attach_course_evidence(
    recommendations_df: pd.DataFrame,
    coursework_df: pd.DataFrame,
    anchor_courses: list[str] | None = None,
) -> pd.DataFrame:
    if recommendations_df.empty:
        return recommendations_df.copy()

    evidence_rows = []
    for course_number in recommendations_df["course_number"]:
        evidence = summarize_course_evidence(coursework_df, course_number, anchor_courses=anchor_courses)
        evidence_rows.append({"course_number": normalize_course_number(course_number), **evidence})

    evidence_df = pd.DataFrame(evidence_rows)
    merged = recommendations_df.merge(evidence_df, on="course_number", how="left")
    return merged


def build_bundle_course_evidence(coursework_df: pd.DataFrame, course_numbers: list[str]) -> pd.DataFrame:
    selected_courses = [normalize_course_number(course) for course in course_numbers if normalize_course_number(course)]
    if not selected_courses:
        return pd.DataFrame(
            columns=["course_number", "bundle_top_partner", "bundle_partner_count", "bundle_historical_support"]
        )

    bundles = build_student_semester_bundles(coursework_df)
    pair_counter: Counter[tuple[str, str]] = Counter()
    for row in bundles.itertuples():
        present = sorted(set(row.courses_taken_together) & set(selected_courses))
        if len(present) < 2:
            continue
        for course_a, course_b in combinations(present, 2):
            pair_counter[(course_a, course_b)] += 1

    rows = []
    for course in selected_courses:
        partner_counts = []
        for partner in selected_courses:
            if partner == course:
                continue
            key = tuple(sorted((course, partner)))
            count = int(pair_counter.get(key, 0))
            if count > 0:
                partner_counts.append((partner, count))

        if partner_counts:
            partner_counts.sort(key=lambda item: (-item[1], item[0]))
            top_partner, top_count = partner_counts[0]
        else:
            top_partner, top_count = None, 0

        rows.append(
            {
                "course_number": course,
                "bundle_top_partner": top_partner,
                "bundle_partner_count": top_count,
                "bundle_historical_support": "Common bundle pairing" if top_count > 0 else "Standalone fit",
            }
        )

    return pd.DataFrame(rows)
