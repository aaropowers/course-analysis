from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .audit import audit_degree_progress, get_eligible_courses
from .features import (
    attach_course_evidence,
    build_coenrollment_features,
    build_student_course_matrix,
    compute_course_similarity,
    interest_overlap,
)
from .utils import (
    build_course_lookup,
    load_catalog,
    load_coursework_records,
    load_degree_plan,
    load_transcripts,
    normalize_course_number,
    split_allowed_courses,
)


DEFAULT_WEIGHTS = {
    "requirement_priority": 0.35,
    "semester_fit": 0.15,
    "similarity": 0.15,
    "coenrollment": 0.10,
    "collaborative": 0.15,
    "interest_match": 0.10,
    "has_corequisite": 0.05,
}


def _requirement_priority_map(audit_result: dict, degree_plan_df: pd.DataFrame) -> dict[str, float]:
    remaining = audit_result["requirements"].query("status == 'Remaining'")
    scores: dict[str, float] = {}
    for _, row in remaining.iterrows():
        match = degree_plan_df[degree_plan_df["requirement_id"] == row["requirement_id"]].iloc[0]
        for course in split_allowed_courses(match["allowed_courses"]):
            scores[course] = max(scores.get(course, 0.0), 1.0)
    return scores


def _semester_fit_score(course_semester: int | float, completed_count: int) -> float:
    estimated_stage = max(1, min(8, int((completed_count / 4) + 1)))
    distance = abs(float(course_semester) - estimated_stage)
    return max(0.0, 1 - (distance / 6))


def _similarity_scores(completed_courses: list[str], similarity_matrix: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    for course in similarity_matrix.index:
        if course in completed_courses:
            continue
        comparisons = [similarity_matrix.loc[taken, course] for taken in completed_courses if taken in similarity_matrix.index]
        scores[course] = float(max(comparisons)) if comparisons else 0.0
    return scores


def _coenrollment_scores(completed_courses: list[str], pair_df: pd.DataFrame) -> dict[str, float]:
    if pair_df.empty:
        return {}
    score_counter: Counter[str] = Counter()
    anchors = set(completed_courses)
    for _, row in pair_df.iterrows():
        course_a = row["course_a"]
        course_b = row["course_b"]
        if course_a in anchors and course_b not in anchors:
            score_counter[course_b] += int(row["coenrollment_count"])
        if course_b in anchors and course_a not in anchors:
            score_counter[course_a] += int(row["coenrollment_count"])

    max_score = max(score_counter.values(), default=0)
    if max_score == 0:
        return {}
    return {course: score / max_score for course, score in score_counter.items()}


def _collaborative_scores(completed_courses: list[str], transcript_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    matrix = build_student_course_matrix(transcript_df)
    if matrix.empty:
        return {}, {}

    profile = pd.Series(0.0, index=matrix.columns, dtype=float)
    for course in completed_courses:
        if course in profile.index:
            profile.loc[course] = 4.0

    profile_frame = profile.to_frame().T
    similarities = cosine_similarity(profile_frame, matrix.fillna(0.0))[0]

    course_support: defaultdict[str, float] = defaultdict(float)
    course_grade_sum: defaultdict[str, float] = defaultdict(float)
    course_weight_sum: defaultdict[str, float] = defaultdict(float)

    for similarity, (_, row) in zip(similarities, matrix.iterrows()):
        if similarity <= 0:
            continue
        for course_number, grade_points in row.dropna().items():
            if course_number in completed_courses:
                continue
            course_support[course_number] += similarity
            course_grade_sum[course_number] += similarity * float(grade_points)
            course_weight_sum[course_number] += similarity

    max_support = max(course_support.values(), default=0.0)
    normalized_support = {
        course: (support / max_support) if max_support else 0.0
        for course, support in course_support.items()
    }
    predicted_gpa = {
        course: round(course_grade_sum[course] / course_weight_sum[course], 2)
        for course in course_grade_sum
        if course_weight_sum[course] > 0
    }
    return normalized_support, predicted_gpa


def explain_recommendation(course: str, student_profile: dict, signals: dict[str, float], evidence: dict[str, object] | None = None) -> str:
    reasons: list[str] = []
    if signals.get("requirement_priority", 0) > 0:
        reasons.append("satisfies a remaining degree requirement")
    if signals.get("semester_fit", 0) >= 0.7:
        reasons.append("fits the typical sequencing for your current stage")
    if signals.get("similarity", 0) >= 0.2:
        reasons.append("is similar to courses you have already completed")
    if signals.get("coenrollment", 0) > 0:
        reasons.append("commonly appears with courses like the ones already on your record")
    if signals.get("interest_match", 0) > 0:
        reasons.append("matches your selected interest areas")
    if signals.get("has_corequisite", 0) > 0:
        reasons.append("can be scheduled now as long as its corequisite is taken in the same term")
    if evidence:
        if evidence.get("evidence_anchor_course") and int(evidence.get("evidence_after_anchor_count", 0) or 0) > 0:
            reasons.append(f"commonly follows {evidence['evidence_anchor_course']} in the coursework dataset")
        if evidence.get("evidence_pass_rate") is not None and float(evidence["evidence_pass_rate"]) >= 90:
            reasons.append("has a strong pass rate in the bootstrapped sample")
        if evidence.get("evidence_top_companion"):
            reasons.append(f"is often taken alongside {evidence['evidence_top_companion']}")

    if not reasons:
        reasons.append("is unlocked and keeps you moving through the plan")

    lookup = build_course_lookup()
    title = lookup.get(course, {}).get("course_title", "")
    lead = f"{course} ({title})" if title else course
    return f"{lead} is recommended because it " + ", ".join(reasons) + "."


def recommend_courses(student_profile: dict, eligible_courses: pd.DataFrame | None = None, models: dict | None = None, weights: dict | None = None) -> pd.DataFrame:
    catalog_df = load_catalog()
    degree_plan_df = load_degree_plan()
    transcript_df = student_profile["transcript_df"]
    weights = weights or DEFAULT_WEIGHTS
    models = models or {
        "similarity": compute_course_similarity(catalog_df),
        "coenrollment": build_coenrollment_features(load_transcripts()),
        "historical_transcripts": load_transcripts(),
    }

    audit_result = audit_degree_progress(transcript_df, degree_plan_df)
    completed_courses = audit_result["completed_courses"]
    eligible_courses = eligible_courses if eligible_courses is not None else get_eligible_courses(transcript_df)
    eligible_courses = eligible_courses.query("is_eligible").copy()

    requirement_scores = _requirement_priority_map(audit_result, degree_plan_df)
    similarity_scores = _similarity_scores(completed_courses, models["similarity"])
    coenrollment_scores = _coenrollment_scores(completed_courses, models["coenrollment"])
    collaborative_scores, predicted_gpa = _collaborative_scores(completed_courses, models["historical_transcripts"])

    rec_rows = []
    for _, course in eligible_courses.iterrows():
        course_number = course["course_number"]
        interest_score = interest_overlap(
            catalog_df.loc[catalog_df["course_number"] == course_number, "interest_tags"].iloc[0],
            student_profile.get("interests", []),
        )
        raw_signals = {
            "requirement_priority": requirement_scores.get(course_number, 0.0),
            "semester_fit": _semester_fit_score(course["recommended_semester"], len(completed_courses)),
            "similarity": similarity_scores.get(course_number, 0.0),
            "coenrollment": coenrollment_scores.get(course_number, 0.0),
            "collaborative": collaborative_scores.get(course_number, 0.0),
            "interest_match": min(1.0, interest_score / max(1, len(student_profile.get("interests", [])))),
            "has_corequisite": 1.0 if str(course.get("missing_coreqs", "")).strip() else 0.0,
        }
        score = sum(weights.get(name, 0.0) * value for name, value in raw_signals.items())
        rec_rows.append(
            {
                "course_number": course_number,
                "course_title": course["course_title"],
                "credits": course["credits"],
                "eligibility_status": course.get("eligibility_status", "Eligible"),
                "missing_prereqs": course.get("missing_prereqs", ""),
                "missing_coreqs": course.get("missing_coreqs", ""),
                "score": round(score, 3),
                "predicted_gpa": predicted_gpa.get(course_number),
                "explanation": explain_recommendation(course_number, student_profile, raw_signals),
                **raw_signals,
            }
        )

    recommendations = pd.DataFrame(rec_rows).sort_values(["score", "predicted_gpa"], ascending=[False, False], na_position="last").reset_index(drop=True)
    coursework_df = load_coursework_records("bootstrapped")
    recommendations = attach_course_evidence(recommendations, coursework_df, anchor_courses=completed_courses)
    recommendations["explanation"] = recommendations.apply(
        lambda row: explain_recommendation(
            row["course_number"],
            student_profile,
            {
                "requirement_priority": row.get("requirement_priority", 0.0),
                "semester_fit": row.get("semester_fit", 0.0),
                "similarity": row.get("similarity", 0.0),
                "coenrollment": row.get("coenrollment", 0.0),
                "collaborative": row.get("collaborative", 0.0),
                "interest_match": row.get("interest_match", 0.0),
                "has_corequisite": row.get("has_corequisite", 0.0),
            },
            {
                "evidence_avg_gpa": row.get("evidence_avg_gpa"),
                "evidence_pass_rate": row.get("evidence_pass_rate"),
                "evidence_records": row.get("evidence_records"),
                "evidence_top_companion": row.get("evidence_top_companion"),
                "evidence_top_companion_count": row.get("evidence_top_companion_count"),
                "evidence_anchor_course": row.get("evidence_anchor_course"),
                "evidence_after_anchor_count": row.get("evidence_after_anchor_count"),
                "evidence_share_after_anchor": row.get("evidence_share_after_anchor"),
                "evidence_difficulty": row.get("evidence_difficulty"),
            },
        ),
        axis=1,
    )

    target_credit_load = int(student_profile.get("target_credit_load", 12) or 12)
    if target_credit_load <= 9:
        return recommendations.head(6)
    if target_credit_load <= 12:
        return recommendations.head(8)
    return recommendations.head(10)
