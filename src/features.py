from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_catalog, load_prereqs, load_transcripts, normalize_course_number, transcript_gpa

GRADE_ORDER = ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]
SEMESTER_ORDER = {"SPRING": 1, "SUMMER": 2, "FALL": 3}
_GRADE_SUMMARY_CACHE: dict[int, pd.DataFrame] = {}
_COMPANION_COUNT_CACHE: dict[int, Counter[tuple[str, str]]] = {}


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


def build_student_grade_history_vector(transcript_df: pd.DataFrame) -> pd.Series:
    graded = filter_letter_grade_records(transcript_df)
    if graded.empty:
        return pd.Series(dtype=float)
    history = (
        graded.groupby("course_number")["grade_points"]
        .mean()
        .sort_index()
    )
    return history.astype(float)


def _grade_points_to_letter(grade_points: float | None) -> str | None:
    if grade_points is None or pd.isna(grade_points):
        return None

    closest_grade = None
    closest_distance = None
    for grade in GRADE_ORDER:
        points = {
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
        }[grade]
        distance = abs(float(grade_points) - points)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_grade = grade
    return closest_grade


def _student_course_terms(coursework_df: pd.DataFrame) -> pd.DataFrame:
    course_terms = coursework_df[["student_id", "course_number", "year", "semester"]].copy()
    course_terms["term_code"] = _term_code(course_terms)
    return (
        course_terms.groupby(["student_id", "course_number"], as_index=False)["term_code"]
        .min()
    )


def _build_history_with_terms(coursework_df: pd.DataFrame) -> pd.DataFrame:
    graded = filter_letter_grade_records(coursework_df)
    if graded.empty:
        return pd.DataFrame(columns=["student_id", "course_number", "grade_points", "grade", "term_code"])

    history_terms = _student_course_terms(graded)
    grade_history = (
        graded.groupby(["student_id", "course_number"], as_index=False)
        .agg(grade_points=("grade_points", "mean"), grade=("grade", "first"))
    )
    return grade_history.merge(history_terms, on=["student_id", "course_number"], how="left")


def find_similar_histories(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    eligible_target_courses: list[str] | pd.Series,
    min_history_overlap: int = 2,
    top_k: int = 80,
    history_with_terms: pd.DataFrame | None = None,
) -> pd.DataFrame:
    student_history = build_student_grade_history_vector(transcript_df)
    normalized_targets = [normalize_course_number(course) for course in list(eligible_target_courses) if normalize_course_number(course)]
    if student_history.empty or not normalized_targets:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    history_with_terms = history_with_terms if history_with_terms is not None else _build_history_with_terms(coursework_df)
    if history_with_terms.empty:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    target_rows = history_with_terms[history_with_terms["course_number"].isin(normalized_targets)].copy()
    if target_rows.empty:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    completed_courses = set(student_history.index.tolist())
    student_history_frame = student_history.rename("student_grade_points").reset_index()

    target_rows = target_rows[
        ["student_id", "course_number", "grade_points", "grade", "term_code"]
    ].rename(
        columns={
            "course_number": "target_course",
            "grade_points": "target_grade_points",
            "grade": "target_grade",
            "term_code": "target_term_code",
        }
    )
    prior_rows = history_with_terms[history_with_terms["course_number"].isin(completed_courses)][
        ["student_id", "course_number", "grade_points", "term_code"]
    ].copy()
    prior_rows = prior_rows.merge(
        target_rows,
        on="student_id",
        how="inner",
    )
    prior_rows = prior_rows[prior_rows["term_code"] < prior_rows["target_term_code"]].copy()
    if prior_rows.empty:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    prior_rows = prior_rows.merge(student_history_frame, on="course_number", how="inner")
    if prior_rows.empty:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    prior_rows["abs_grade_diff"] = (prior_rows["grade_points"] - prior_rows["student_grade_points"]).abs()
    similar_df = (
        prior_rows.groupby(["target_course", "student_id", "target_grade_points", "target_grade"], as_index=False)
        .agg(
            overlap_count=("course_number", "nunique"),
            mean_abs_grade_diff=("abs_grade_diff", "mean"),
        )
    )
    similar_df = similar_df[similar_df["overlap_count"] >= int(min_history_overlap)].copy()
    if similar_df.empty:
        return pd.DataFrame(
            columns=[
                "target_course",
                "student_id",
                "similarity_score",
                "overlap_count",
                "overlap_strength",
                "mean_abs_grade_diff",
                "target_grade_points",
                "target_grade",
            ]
        )

    similar_df["overlap_strength"] = similar_df["overlap_count"] / max(len(student_history), 1)
    similar_df["similarity_score"] = similar_df["overlap_count"] * (1 / (1 + similar_df["mean_abs_grade_diff"]))
    similar_df = similar_df[similar_df["similarity_score"] > 0].copy()
    similar_df["similarity_score"] = similar_df["similarity_score"].round(4)
    similar_df["overlap_strength"] = similar_df["overlap_strength"].round(4)
    similar_df["mean_abs_grade_diff"] = similar_df["mean_abs_grade_diff"].round(4)
    return (
        similar_df.sort_values(["target_course", "similarity_score", "overlap_count"], ascending=[True, False, False])
        .groupby("target_course", group_keys=False)
        .head(int(top_k))
        .reset_index(drop=True)
    )


def _build_prediction_driver_summary(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    target_course: str,
    matched_neighbors: pd.DataFrame,
    history_with_terms: pd.DataFrame | None = None,
) -> list[str]:
    student_history = build_student_grade_history_vector(transcript_df)
    if student_history.empty or matched_neighbors.empty:
        return []

    history_with_terms = history_with_terms if history_with_terms is not None else _build_history_with_terms(coursework_df)
    if history_with_terms.empty:
        return []
    target_terms = history_with_terms[history_with_terms["course_number"] == normalize_course_number(target_course)][["student_id", "term_code"]]

    prior = history_with_terms.merge(target_terms, on="student_id", how="inner", suffixes=("", "_target"))
    prior = prior[prior["term_code"] < prior["term_code_target"]]
    prior = prior[prior["student_id"].isin(matched_neighbors["student_id"])]
    prior = prior[prior["course_number"].isin(student_history.index)]
    if prior.empty:
        return []

    prior = prior.merge(
        matched_neighbors[["student_id", "similarity_score"]],
        on="student_id",
        how="left",
    )
    prior = prior.merge(
        student_history.rename("student_grade_points").reset_index(),
        on="course_number",
        how="left",
    )
    prior["abs_diff"] = (prior["grade_points"] - prior["student_grade_points"]).abs()
    driver_df = (
        prior.groupby("course_number", as_index=False)
        .agg(
            support=("student_id", "nunique"),
            avg_abs_diff=("abs_diff", "mean"),
            weighted_similarity=("similarity_score", "sum"),
        )
        .sort_values(["support", "weighted_similarity", "avg_abs_diff"], ascending=[False, False, True])
        .head(3)
    )
    return driver_df["course_number"].tolist()


def _build_anchor_evidence(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    target_course: str,
    history_with_terms: pd.DataFrame | None = None,
) -> dict[str, object] | None:
    student_history = build_student_grade_history_vector(transcript_df)
    if student_history.empty:
        return None

    prereq_df = load_prereqs()
    prereq_anchors = prereq_df[
        (prereq_df["course_number"] == normalize_course_number(target_course))
        & (prereq_df["prereq_type"] == "PREREQ")
    ]["prerequisite_course"].tolist()
    candidate_anchors = [course for course in prereq_anchors if course in student_history.index]
    if not candidate_anchors:
        return None

    anchor_course = candidate_anchors[0]
    student_anchor_grade_points = float(student_history[anchor_course])
    history_with_terms = history_with_terms if history_with_terms is not None else _build_history_with_terms(coursework_df)
    if history_with_terms.empty:
        return None
    anchor_rows = history_with_terms[history_with_terms["course_number"] == anchor_course][["student_id", "grade_points", "grade", "term_code"]].rename(
        columns={"grade_points": "anchor_grade_points", "grade": "anchor_grade", "term_code": "anchor_term_code"}
    )
    target_rows = history_with_terms[history_with_terms["course_number"] == normalize_course_number(target_course)][["student_id", "grade_points", "grade", "term_code"]].rename(
        columns={"grade_points": "target_grade_points", "grade": "target_grade", "term_code": "target_term_code"}
    )
    joined = anchor_rows.merge(target_rows, on="student_id", how="inner")
    if joined.empty:
        return None
    joined = joined[joined["anchor_term_code"] < joined["target_term_code"]].copy()
    if joined.empty:
        return None

    tolerance = 0.35
    peer_anchor = joined[(joined["anchor_grade_points"] - student_anchor_grade_points).abs() <= tolerance].copy()
    if len(peer_anchor) < 5:
        return None

    similar_anchor_avg = round(float(peer_anchor["target_grade_points"].mean()), 2)
    return {
        "anchor_course": anchor_course,
        "student_anchor_grade": _grade_points_to_letter(student_anchor_grade_points),
        "student_anchor_grade_points": round(student_anchor_grade_points, 2),
        "similar_anchor_avg_gpa": similar_anchor_avg,
        "similar_anchor_avg_letter": _grade_points_to_letter(similar_anchor_avg),
        "similar_anchor_sample_size": int(len(peer_anchor)),
    }


def _predict_from_neighbors(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    target_course: str,
    neighbors: pd.DataFrame,
    min_matches: int,
    history_with_terms: pd.DataFrame | None = None,
) -> dict[str, object]:
    target_course = normalize_course_number(target_course)
    base_payload = {
        "target_course": target_course,
        "predicted_gpa": None,
        "predicted_letter_grade": None,
        "prediction_low_gpa": None,
        "prediction_high_gpa": None,
        "matched_student_count": 0,
        "history_overlap_strength": None,
        "anchor_evidence": None,
        "explanation": None,
        "status": "empty",
        "empty_reason": None,
    }

    student_history = build_student_grade_history_vector(transcript_df)
    if student_history.empty:
        base_payload["empty_reason"] = "no_student_grade_history"
        return base_payload

    neighbors = neighbors[neighbors["target_course"] == target_course].copy()
    if neighbors.empty or len(neighbors) < int(min_matches):
        base_payload["empty_reason"] = "insufficient_history_matches"
        return base_payload

    weights = neighbors["similarity_score"].astype(float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        base_payload["empty_reason"] = "zero_similarity_weight"
        return base_payload

    predicted_gpa = float((neighbors["target_grade_points"] * weights).sum() / total_weight)
    prediction_low = float(neighbors["target_grade_points"].quantile(0.25))
    prediction_high = float(neighbors["target_grade_points"].quantile(0.75))
    overlap_strength = float((neighbors["overlap_strength"] * weights).sum() / total_weight)
    driver_courses = _build_prediction_driver_summary(
        coursework_df,
        transcript_df,
        target_course,
        neighbors,
        history_with_terms=history_with_terms,
    )
    anchor_evidence = _build_anchor_evidence(
        coursework_df,
        transcript_df,
        target_course,
        history_with_terms=history_with_terms,
    )

    explanation_parts = [
        f"Students with transcript histories most similar to yours averaged {_grade_points_to_letter(predicted_gpa)} ({predicted_gpa:.2f}) in {target_course}."
    ]
    if driver_courses:
        explanation_parts.append(f"The strongest prior-course matches came from {', '.join(driver_courses)}.")
    if anchor_evidence:
        explanation_parts.append(
            f"{anchor_evidence['anchor_course']} students near your {anchor_evidence['student_anchor_grade']} averaged "
            f"{anchor_evidence['similar_anchor_avg_letter']} in {target_course}."
        )

    base_payload.update(
        {
            "predicted_gpa": round(predicted_gpa, 2),
            "predicted_letter_grade": _grade_points_to_letter(predicted_gpa),
            "prediction_low_gpa": round(prediction_low, 2),
            "prediction_high_gpa": round(prediction_high, 2),
            "matched_student_count": int(len(neighbors)),
            "history_overlap_strength": round(overlap_strength, 2),
            "anchor_evidence": anchor_evidence,
            "explanation": " ".join(explanation_parts),
            "status": "ok",
            "empty_reason": None,
        }
    )
    return base_payload


def predict_future_course_performance(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    target_course: str,
    min_history_overlap: int = 2,
    min_matches: int = 8,
    top_k: int = 80,
) -> dict[str, object]:
    target_course = normalize_course_number(target_course)
    history_with_terms = _build_history_with_terms(coursework_df)
    neighbors = find_similar_histories(
        coursework_df,
        transcript_df,
        eligible_target_courses=[target_course],
        min_history_overlap=min_history_overlap,
        top_k=top_k,
        history_with_terms=history_with_terms,
    )
    return _predict_from_neighbors(
        coursework_df,
        transcript_df,
        target_course=target_course,
        neighbors=neighbors,
        min_matches=min_matches,
        history_with_terms=history_with_terms,
    )


def predict_eligible_course_performance(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    eligible_courses: pd.DataFrame,
    min_history_overlap: int = 2,
    min_matches: int = 8,
    top_k: int = 80,
) -> pd.DataFrame:
    if eligible_courses.empty:
        return pd.DataFrame(
            columns=[
                "course_number",
                "predicted_gpa",
                "predicted_letter_grade",
                "prediction_low_gpa",
                "prediction_high_gpa",
                "prediction_sample_size",
                "prediction_confidence",
                "prediction_evidence_level",
                "prediction_explanation",
                "history_overlap_strength",
                "anchor_evidence",
                "prediction_status",
                "prediction_empty_reason",
            ]
        )

    target_courses = [normalize_course_number(course) for course in eligible_courses["course_number"].dropna().tolist()]
    history_with_terms = _build_history_with_terms(coursework_df)
    neighbors = find_similar_histories(
        coursework_df,
        transcript_df,
        eligible_target_courses=target_courses,
        min_history_overlap=min_history_overlap,
        top_k=top_k,
        history_with_terms=history_with_terms,
    )

    rows = []
    for course_number in target_courses:
        payload = _predict_from_neighbors(
            coursework_df,
            transcript_df,
            target_course=course_number,
            neighbors=neighbors,
            min_matches=min_matches,
            history_with_terms=history_with_terms,
        )
        sample_size = int(payload["matched_student_count"] or 0)
        if payload["status"] == "ok":
            if sample_size >= 35:
                evidence_level = "High evidence"
            elif sample_size >= 18:
                evidence_level = "Moderate evidence"
            else:
                evidence_level = "Low evidence"
            confidence = round(min(1.0, (sample_size / 40)) * float(payload["history_overlap_strength"] or 0), 2)
        else:
            evidence_level = "Low evidence"
            confidence = 0.0

        rows.append(
            {
                "course_number": payload["target_course"],
                "predicted_gpa": payload["predicted_gpa"],
                "predicted_letter_grade": payload["predicted_letter_grade"],
                "prediction_low_gpa": payload["prediction_low_gpa"],
                "prediction_high_gpa": payload["prediction_high_gpa"],
                "prediction_sample_size": sample_size,
                "prediction_confidence": confidence,
                "prediction_evidence_level": evidence_level,
                "prediction_explanation": payload["explanation"],
                "history_overlap_strength": payload["history_overlap_strength"],
                "anchor_evidence": payload["anchor_evidence"],
                "prediction_status": payload["status"],
                "prediction_empty_reason": payload["empty_reason"],
            }
        )

    return pd.DataFrame(rows)


def build_peer_cohort(coursework_df: pd.DataFrame, student_gpa: float | None, gpa_band: float = 0.15) -> pd.DataFrame:
    graded = filter_letter_grade_records(coursework_df)
    if graded.empty or student_gpa is None:
        return graded.iloc[0:0].copy()

    student_baselines = (
        graded.groupby("student_id", as_index=False)["grade_points"]
        .mean()
        .rename(columns={"grade_points": "student_gpa"})
    )
    cohort_students = student_baselines[
        student_baselines["student_gpa"].between(student_gpa - gpa_band, student_gpa + gpa_band, inclusive="both")
    ].copy()
    if cohort_students.empty:
        return graded.iloc[0:0].copy()

    cohort = graded.merge(cohort_students, on="student_id", how="inner")
    cohort["gpa_delta"] = cohort["grade_points"] - cohort["student_gpa"]
    return cohort


def build_student_normalized_course_profile(transcript_df: pd.DataFrame) -> pd.DataFrame:
    graded = filter_letter_grade_records(transcript_df)
    student_gpa = transcript_gpa(transcript_df)
    if graded.empty or student_gpa is None:
        return pd.DataFrame(columns=["course_number", "grade", "grade_points", "student_gpa", "gpa_delta"])

    profile = graded[["course_number", "grade", "grade_points"]].copy()
    profile["student_gpa"] = student_gpa
    profile["gpa_delta"] = profile["grade_points"] - student_gpa
    return profile.sort_values(["gpa_delta", "course_number"], ascending=[True, True]).reset_index(drop=True)


def _percentile_less_equal(values: pd.Series, target: float | None) -> float | None:
    if target is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    percentile = float((numeric <= target).mean() * 100)
    return round(percentile, 1)


def build_peer_normalized_course_metrics(
    coursework_df: pd.DataFrame,
    student_gpa: float | None,
    min_sample: int = 15,
    gpa_band: float = 0.15,
) -> pd.DataFrame:
    cohort = build_peer_cohort(coursework_df, student_gpa=student_gpa, gpa_band=gpa_band)
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "course_number",
                "peer_sample_size",
                "peer_student_count",
                "peer_avg_grade_points",
                "peer_avg_gpa_delta",
                "peer_grade_points_std",
                "peer_distribution_spread",
            ]
        )

    metrics = (
        cohort.groupby("course_number")
        .agg(
            peer_sample_size=("grade_points", "size"),
            peer_student_count=("student_id", "nunique"),
            peer_avg_grade_points=("grade_points", "mean"),
            peer_avg_gpa_delta=("gpa_delta", "mean"),
            peer_grade_points_std=("grade_points", "std"),
        )
        .reset_index()
    )
    metrics = metrics[metrics["peer_sample_size"] >= int(min_sample)].copy()
    if metrics.empty:
        return metrics

    metrics["peer_avg_grade_points"] = metrics["peer_avg_grade_points"].round(2)
    metrics["peer_avg_gpa_delta"] = metrics["peer_avg_gpa_delta"].round(2)
    metrics["peer_grade_points_std"] = metrics["peer_grade_points_std"].fillna(0).round(2)
    metrics["peer_distribution_spread"] = metrics["peer_grade_points_std"]
    return metrics.sort_values(["peer_sample_size", "course_number"], ascending=[False, True]).reset_index(drop=True)


def build_anchor_course_recommendation(
    transcript_df: pd.DataFrame,
    peer_metrics_df: pd.DataFrame,
    min_sample: int = 15,
) -> dict:
    profile_df = build_student_normalized_course_profile(transcript_df)
    if profile_df.empty:
        return {
            "anchor_course": None,
            "eligible_anchor_courses": [],
            "anchor_candidates_df": pd.DataFrame(),
            "status": "empty",
            "empty_reason": "no_student_gpa",
        }

    if peer_metrics_df.empty:
        return {
            "anchor_course": None,
            "eligible_anchor_courses": [],
            "anchor_candidates_df": pd.DataFrame(),
            "status": "empty",
            "empty_reason": "no_peer_courses",
        }

    candidates = profile_df.merge(peer_metrics_df, on="course_number", how="inner")
    candidates = candidates[candidates["peer_sample_size"] >= int(min_sample)].copy()
    if candidates.empty:
        return {
            "anchor_course": None,
            "eligible_anchor_courses": [],
            "anchor_candidates_df": pd.DataFrame(),
            "status": "empty",
            "empty_reason": "no_valid_anchor_course",
        }

    candidates = candidates.sort_values(
        ["gpa_delta", "peer_sample_size", "course_number"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    eligible_anchor_courses = candidates["course_number"].drop_duplicates().tolist()
    return {
        "anchor_course": eligible_anchor_courses[0] if eligible_anchor_courses else None,
        "eligible_anchor_courses": eligible_anchor_courses,
        "anchor_candidates_df": candidates,
        "status": "ok",
        "empty_reason": None,
    }


def build_peer_anchor_insight(
    coursework_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    anchor_course: str | None,
    gpa_band: float = 0.15,
    min_sample: int = 15,
) -> dict:
    student_gpa = transcript_gpa(transcript_df)
    base_payload = {
        "student_gpa": student_gpa,
        "peer_student_count": 0,
        "anchor_course": normalize_course_number(anchor_course),
        "anchor_student_grade": None,
        "anchor_peer_distribution_df": pd.DataFrame(columns=["grade", "count", "cumulative_pct", "student_grade_flag"]),
        "anchor_summary": {},
        "comparison_courses_df": pd.DataFrame(),
        "eligible_anchor_courses": [],
        "status": "empty",
        "empty_reason": None,
    }
    if student_gpa is None:
        base_payload["empty_reason"] = "no_student_gpa"
        return base_payload

    peer_metrics_df = build_peer_normalized_course_metrics(
        coursework_df,
        student_gpa=student_gpa,
        min_sample=min_sample,
        gpa_band=gpa_band,
    )
    anchor_choice = build_anchor_course_recommendation(transcript_df, peer_metrics_df, min_sample=min_sample)
    base_payload["eligible_anchor_courses"] = anchor_choice["eligible_anchor_courses"]
    if anchor_choice["status"] != "ok":
        base_payload["empty_reason"] = anchor_choice["empty_reason"]
        return base_payload

    normalized_anchor = normalize_course_number(anchor_course) if anchor_course else anchor_choice["anchor_course"]
    if normalized_anchor not in anchor_choice["eligible_anchor_courses"]:
        normalized_anchor = anchor_choice["anchor_course"]
    if not normalized_anchor:
        base_payload["empty_reason"] = "no_valid_anchor_course"
        return base_payload

    cohort = build_peer_cohort(coursework_df, student_gpa=student_gpa, gpa_band=gpa_band)
    if cohort.empty:
        base_payload["empty_reason"] = "no_peer_cohort"
        return base_payload

    base_payload["peer_student_count"] = int(cohort["student_id"].nunique())
    base_payload["anchor_course"] = normalized_anchor

    profile_df = build_student_normalized_course_profile(transcript_df)
    student_anchor = profile_df[profile_df["course_number"] == normalized_anchor].head(1)
    if student_anchor.empty:
        base_payload["empty_reason"] = "anchor_not_completed"
        return base_payload

    student_anchor_grade_points = float(student_anchor.iloc[0]["grade_points"])
    student_anchor_grade = student_anchor.iloc[0]["grade"]
    student_anchor_delta = float(student_anchor.iloc[0]["gpa_delta"])
    base_payload["anchor_student_grade"] = student_anchor_grade

    anchor_peer = cohort[cohort["course_number"] == normalized_anchor].copy()
    if len(anchor_peer) < int(min_sample):
        base_payload["empty_reason"] = "anchor_sparse"
        return base_payload

    anchor_distribution = (
        anchor_peer["grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    )
    anchor_distribution = anchor_distribution[anchor_distribution > 0]
    anchor_distribution_df = pd.DataFrame(
        {
            "grade": anchor_distribution.index,
            "count": anchor_distribution.values,
        }
    )
    anchor_distribution_df["cumulative_pct"] = (
        anchor_distribution_df["count"].cumsum() / anchor_distribution_df["count"].sum() * 100
    ).round(1)
    anchor_distribution_df["student_grade_flag"] = anchor_distribution_df["grade"].eq(student_anchor_grade)

    comparison_df = profile_df.merge(peer_metrics_df, on="course_number", how="inner")
    comparison_df = comparison_df[comparison_df["course_number"] != normalized_anchor].copy()
    if not comparison_df.empty:
        peer_percentiles = []
        for row in comparison_df.itertuples():
            course_peer = cohort[cohort["course_number"] == row.course_number]
            peer_percentiles.append(_percentile_less_equal(course_peer["grade_points"], float(row.grade_points)))
        comparison_df["peer_percentile"] = peer_percentiles
        comparison_df["delta_vs_peer_avg"] = (comparison_df["gpa_delta"] - comparison_df["peer_avg_gpa_delta"]).round(2)
        comparison_df["grade_gap_vs_peer_avg"] = (comparison_df["grade_points"] - comparison_df["peer_avg_grade_points"]).round(2)
        comparison_df = comparison_df.sort_values(
            ["peer_sample_size", "peer_percentile", "gpa_delta", "course_number"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        comparison_df = comparison_df.head(8)

    anchor_percentile = _percentile_less_equal(anchor_peer["grade_points"], student_anchor_grade_points)
    peer_avg_grade_points = round(float(anchor_peer["grade_points"].mean()), 2)
    peer_avg_gpa_delta = round(float(anchor_peer["gpa_delta"].mean()), 2)
    delta_vs_peer_avg = round(student_anchor_delta - peer_avg_gpa_delta, 2)
    grade_gap_vs_peer_avg = round(student_anchor_grade_points - peer_avg_grade_points, 2)

    base_payload["anchor_peer_distribution_df"] = anchor_distribution_df
    base_payload["anchor_summary"] = {
        "peer_sample_size": int(len(anchor_peer)),
        "peer_avg_grade_points": peer_avg_grade_points,
        "peer_avg_gpa_delta": peer_avg_gpa_delta,
        "student_gpa_delta": round(student_anchor_delta, 2),
        "peer_percentile": anchor_percentile,
        "delta_vs_peer_avg": delta_vs_peer_avg,
        "grade_gap_vs_peer_avg": grade_gap_vs_peer_avg,
        "underperformed_peer_average": delta_vs_peer_avg < 0,
    }
    base_payload["comparison_courses_df"] = comparison_df
    base_payload["status"] = "ok"
    return base_payload


def interest_overlap(course_interest_tags: str, selected_interests: list[str]) -> int:
    if not selected_interests:
        return 0
    course_tags = {normalize_course_number(tag) for tag in str(course_interest_tags).split("|")}
    desired = {normalize_course_number(tag) for tag in selected_interests}
    return len(course_tags & desired)


def filter_letter_grade_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["grade"].isin(GRADE_ORDER)].copy()


def _cached_grade_summary(coursework_df: pd.DataFrame) -> pd.DataFrame:
    cache_key = id(coursework_df)
    if cache_key not in _GRADE_SUMMARY_CACHE:
        graded = filter_letter_grade_records(coursework_df)
        summary = (
            graded.groupby("course_number")
            .agg(
                evidence_avg_gpa=("grade_points", "mean"),
                evidence_records=("grade_points", "count"),
                evidence_pass_rate=("grade_points", lambda values: (values >= 1.7).mean() * 100),
            )
            .reset_index()
        )
        if summary.empty:
            summary = pd.DataFrame(columns=["course_number", "evidence_avg_gpa", "evidence_records", "evidence_pass_rate"])
        summary["evidence_avg_gpa"] = summary["evidence_avg_gpa"].round(2)
        summary["evidence_pass_rate"] = summary["evidence_pass_rate"].round(1)
        _GRADE_SUMMARY_CACHE[cache_key] = summary
    return _GRADE_SUMMARY_CACHE[cache_key]


def _cached_companion_counts(coursework_df: pd.DataFrame) -> Counter[tuple[str, str]]:
    cache_key = id(coursework_df)
    if cache_key not in _COMPANION_COUNT_CACHE:
        companion_counter: Counter[tuple[str, str]] = Counter()
        bundles = build_student_semester_bundles(coursework_df)
        for row in bundles.itertuples():
            unique_courses = sorted(set(row.courses_taken_together))
            for course_a, course_b in combinations(unique_courses, 2):
                companion_counter[(course_a, course_b)] += 1
                companion_counter[(course_b, course_a)] += 1
        _COMPANION_COUNT_CACHE[cache_key] = companion_counter
    return _COMPANION_COUNT_CACHE[cache_key]


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
    include_sequence: bool = True,
) -> pd.DataFrame:
    if recommendations_df.empty:
        return recommendations_df.copy()

    if not include_sequence:
        target_courses = [
            normalize_course_number(course)
            for course in recommendations_df["course_number"].dropna().unique().tolist()
        ]
        grade_summary = _cached_grade_summary(coursework_df)
        grade_summary = grade_summary[grade_summary["course_number"].isin(target_courses)].copy()
        companion_counter = _cached_companion_counts(coursework_df)

        companion_rows = []
        for target in target_courses:
            companions = [
                (companion, count)
                for (course, companion), count in companion_counter.items()
                if course == target
            ]
            if companions:
                companions.sort(key=lambda item: (-item[1], item[0]))
                top_companion, top_count = companions[0]
            else:
                top_companion, top_count = None, 0
            companion_rows.append(
                {
                    "course_number": target,
                    "evidence_top_companion": top_companion,
                    "evidence_top_companion_count": top_count,
                }
            )

        evidence_df = pd.DataFrame({"course_number": target_courses})
        evidence_df = evidence_df.merge(grade_summary, on="course_number", how="left")
        evidence_df = evidence_df.merge(pd.DataFrame(companion_rows), on="course_number", how="left")
        evidence_df["evidence_avg_gpa"] = evidence_df["evidence_avg_gpa"].where(evidence_df["evidence_avg_gpa"].notna(), None)
        evidence_df["evidence_pass_rate"] = evidence_df["evidence_pass_rate"].where(evidence_df["evidence_pass_rate"].notna(), None)
        evidence_df["evidence_records"] = evidence_df["evidence_records"].fillna(0).astype(int)
        evidence_df["evidence_top_companion_count"] = evidence_df["evidence_top_companion_count"].fillna(0).astype(int)
        evidence_df["evidence_anchor_course"] = None
        evidence_df["evidence_after_anchor_count"] = 0
        evidence_df["evidence_anchor_target_students"] = 0
        evidence_df["evidence_share_after_anchor"] = None
        evidence_df["evidence_difficulty"] = evidence_df.apply(
            lambda row: _difficulty_label(row["evidence_avg_gpa"], row["evidence_pass_rate"]),
            axis=1,
        )
        return recommendations_df.merge(evidence_df, on="course_number", how="left")

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
