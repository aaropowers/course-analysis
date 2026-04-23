from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import get_eligible_courses
from src.recommender import _personalized_gpa_prediction, recommend_courses
from src.utils import load_transcripts


def test_student_gpa_shifts_personalized_prediction() -> None:
    high_gpa_prediction = _personalized_gpa_prediction(
        course_number="ME330",
        student_gpa=3.8,
        historical_avg_gpa=3.2,
        course_avg_gpa=3.0,
        course_std_gpa=0.5,
        evidence_records=120,
        collaborative_gpa=3.1,
    )
    low_gpa_prediction = _personalized_gpa_prediction(
        course_number="ME330",
        student_gpa=2.6,
        historical_avg_gpa=3.2,
        course_avg_gpa=3.0,
        course_std_gpa=0.5,
        evidence_records=120,
        collaborative_gpa=3.1,
    )

    assert high_gpa_prediction["predicted_gpa"] > low_gpa_prediction["predicted_gpa"]


def test_sparse_or_missing_context_widens_prediction_range() -> None:
    strong_context = _personalized_gpa_prediction(
        course_number="ME330",
        student_gpa=3.4,
        historical_avg_gpa=3.2,
        course_avg_gpa=3.0,
        course_std_gpa=0.4,
        evidence_records=120,
        collaborative_gpa=3.1,
    )
    sparse_context = _personalized_gpa_prediction(
        course_number="ME330",
        student_gpa=None,
        historical_avg_gpa=3.2,
        course_avg_gpa=3.0,
        course_std_gpa=0.9,
        evidence_records=5,
        collaborative_gpa=None,
    )

    strong_width = strong_context["predicted_gpa_high"] - strong_context["predicted_gpa_low"]
    sparse_width = sparse_context["predicted_gpa_high"] - sparse_context["predicted_gpa_low"]

    assert sparse_width > strong_width
    assert sparse_context["prediction_confidence"] == "Low"


def test_prediction_outputs_are_clamped_to_gpa_scale() -> None:
    prediction = _personalized_gpa_prediction(
        course_number="ME330",
        student_gpa=4.0,
        historical_avg_gpa=2.0,
        course_avg_gpa=3.9,
        course_std_gpa=0.8,
        evidence_records=3,
        collaborative_gpa=4.0,
    )

    assert prediction["predicted_gpa"] == 4.0
    assert 0.0 <= prediction["predicted_gpa_low"] <= 4.0
    assert prediction["predicted_gpa_high"] == 4.0


def test_recommend_courses_returns_personalized_prediction_columns() -> None:
    transcripts = load_transcripts()
    transcript_df = transcripts[transcripts["student_id"] == "demo_mid"].copy()
    recommendations = recommend_courses(
        {
            "student_id": "demo_mid",
            "transcript_df": transcript_df,
            "interests": ["controls", "robotics"],
            "target_credit_load": 12,
            "student_gpa": 3.36,
        },
        eligible_courses=get_eligible_courses(transcript_df),
    )

    expected_columns = {
        "student_gpa",
        "predicted_gpa",
        "predicted_gpa_low",
        "predicted_gpa_high",
        "prediction_confidence",
        "prediction_basis",
    }
    assert expected_columns.issubset(recommendations.columns)
    assert recommendations["predicted_gpa"].notna().any()


def main() -> int:
    test_student_gpa_shifts_personalized_prediction()
    test_sparse_or_missing_context_widens_prediction_range()
    test_prediction_outputs_are_clamped_to_gpa_scale()
    test_recommend_courses_returns_personalized_prediction_columns()
    print("[PASS] GPA-normalized recommendation tests succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
