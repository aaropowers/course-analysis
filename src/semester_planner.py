from __future__ import annotations

import datetime

import pandas as pd

from .audit import (
    _completed_courses,
    _in_progress_courses,
    build_degree_plan_progression,
    build_transcript_progression,
)
from .features import build_bundle_course_evidence
from .utils import (
    load_catalog,
    load_coursework_records,
    load_prereqs,
    split_allowed_courses,
    term_sort_key,
)


_ROADMAP_COLUMNS = [
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
    "requirement_id",
    "requirement_name",
    "is_elective_slot",
    "allowed_courses",
]

_CATEGORY_RANK = {
    "Math": 1,
    "Science": 2,
    "Computing": 3,
    "General Education": 4,
    "ME Core": 5,
    "ME Lab": 6,
    "Gateway Elective": 7,
    "Other": 8,
}

_SEASON_ORDER = {"SPRING": 1, "SUMMER": 2, "FALL": 3}


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


def _advance_term(year: int, season: str, include_summer: bool) -> tuple[int, str]:
    season = season.upper()
    if season == "SPRING":
        return (year, "SUMMER") if include_summer else (year, "FALL")
    if season == "SUMMER":
        return year, "FALL"
    if season == "FALL":
        return year + 1, "SPRING"
    return year + 1, "SPRING"


def _term_order_value(year: int, season: str) -> int:
    return year * 10 + _SEASON_ORDER.get(season.upper(), 9)


def _latest_transcript_term(transcript_rows: pd.DataFrame) -> tuple[int, str] | None:
    if transcript_rows.empty or "term" not in transcript_rows.columns:
        return None
    candidate = transcript_rows.dropna(subset=["term"]).copy()
    if candidate.empty:
        return None
    def _order_for(value: object) -> int:
        if not isinstance(value, str) or "-" not in value:
            return -1
        try:
            year, season_rank = term_sort_key(value)
        except Exception:
            return -1
        return year * 10 + season_rank

    candidate["_order"] = candidate["term"].map(_order_for)
    candidate = candidate[candidate["_order"] >= 0]
    if candidate.empty:
        return None
    best_row = candidate.loc[candidate["_order"].idxmax()]
    term_value = str(best_row["term"])
    try:
        year_text, season = term_value.split("-", 1)
        return int(year_text), season.upper()
    except ValueError:
        return None


def _default_start_term() -> tuple[int, str]:
    now = datetime.datetime.now()
    if now.month <= 5:
        return now.year, "SPRING"
    if now.month <= 7:
        return now.year, "SUMMER"
    return now.year, "FALL"


def _coerce_credits(value, default: float = 3.0) -> float:
    try:
        credits = float(value)
    except (TypeError, ValueError):
        return default
    return credits if credits > 0 else default


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_remaining_pool(
    remaining_reqs: pd.DataFrame,
    dependency_lookup: dict[str, dict],
    catalog_lookup: dict[str, dict],
    prereq_df: pd.DataFrame,
) -> list[dict]:
    """Turn each remaining requirement into a scheduler item.

    For Gateway Electives the default ``course_number`` emitted by
    :func:`build_degree_plan_progression` may repeat across ELEC-1 / ELEC-2;
    we reassign duplicates to the next available allowed course so each
    elective slot picks a distinct real course.
    """
    pool: list[dict] = []
    used_electives: set[str] = set()

    for _, req_row in remaining_reqs.iterrows():
        allowed = sorted(split_allowed_courses(req_row.get("allowed_courses", "")))
        category = req_row.get("category", "Other")
        is_elective = category == "Gateway Elective"
        primary_course = str(req_row.get("course_number") or "").strip()

        if is_elective:
            chosen = None
            for candidate in allowed:
                if candidate not in used_electives:
                    chosen = candidate
                    break
            primary_course = chosen or primary_course or (allowed[0] if allowed else "")
            if primary_course:
                used_electives.add(primary_course)

        if not primary_course:
            continue

        catalog_meta = catalog_lookup.get(primary_course, {})
        dep_meta = dependency_lookup.get(primary_course, {})

        course_title = (
            catalog_meta.get("course_title")
            or dep_meta.get("course_title")
            or req_row.get("requirement_name")
            or primary_course
        )
        credits = _coerce_credits(
            catalog_meta.get("credits") or dep_meta.get("credits"),
            default=_coerce_credits(req_row.get("required_hours"), default=3.0),
        )

        course_rows = prereq_df[prereq_df["course_number"] == primary_course]
        prereqs = set(
            course_rows.loc[course_rows["prereq_type"] == "PREREQ", "prerequisite_course"]
        )
        coreqs = set(
            course_rows.loc[course_rows["prereq_type"] == "COREQ", "prerequisite_course"]
        )

        pool.append(
            {
                "requirement_id": req_row.get("requirement_id", ""),
                "requirement_name": req_row.get("requirement_name", ""),
                "category": category,
                "allowed_courses": req_row.get("allowed_courses", ""),
                "is_elective_slot": bool(is_elective),
                "course_number": primary_course,
                "course_title": course_title,
                "credits": credits,
                "prereqs": prereqs,
                "coreqs": coreqs,
                "recommended_semester": _coerce_int(
                    req_row.get("recommended_semester"), default=8
                ),
            }
        )

    return pool


def _transcript_rows_with_schema(transcript_rows: pd.DataFrame) -> pd.DataFrame:
    """Ensure transcript rows carry the extra roadmap columns for a uniform schema."""
    enriched = transcript_rows.copy()
    for column in ("requirement_id", "requirement_name", "allowed_courses"):
        if column not in enriched.columns:
            enriched[column] = pd.NA
    if "is_elective_slot" not in enriched.columns:
        enriched["is_elective_slot"] = False
    else:
        enriched["is_elective_slot"] = enriched["is_elective_slot"].fillna(False).astype(bool)
    return enriched[_ROADMAP_COLUMNS]


def build_graduation_roadmap(
    transcript_df: pd.DataFrame,
    target_credits_per_term: int = 12,
    include_summer: bool = False,
    max_future_terms: int = 12,
) -> pd.DataFrame:
    """Combine a student's transcript with a prereq-aware forward schedule.

    Returns a DataFrame that mirrors the columns of
    :func:`src.audit.build_transcript_progression` with four roadmap extras
    (``requirement_id``, ``requirement_name``, ``is_elective_slot``,
    ``allowed_courses``). Past and in-progress courses come from the transcript
    unchanged; remaining degree requirements are placed in future terms using
    a greedy, prereq-aware scheduler that respects ``target_credits_per_term``
    and co-schedules known corequisite pairs. Any requirement the scheduler
    could not place within ``max_future_terms`` is returned with term
    ``"UNSCHEDULED"`` so the UI can surface it separately.
    """
    transcript_rows = build_transcript_progression(transcript_df)
    transcript_rows = _transcript_rows_with_schema(transcript_rows)

    progression_df, _ = build_degree_plan_progression(transcript_df)
    unmet_statuses = {"Eligible", "Eligible with corequisite", "Locked", "Remaining"}
    remaining_reqs = (
        progression_df[progression_df["status"].isin(unmet_statuses)].copy()
        if not progression_df.empty
        else progression_df
    )

    if remaining_reqs is None or remaining_reqs.empty:
        return transcript_rows.reset_index(drop=True)

    catalog_lookup = load_catalog().set_index("course_number").to_dict(orient="index")
    prereq_df = load_prereqs()

    dependency_lookup: dict[str, dict] = {}
    for col in ("course_number", "course_title", "credits"):
        if col not in progression_df.columns:
            break
    else:
        dependency_lookup = (
            progression_df.drop_duplicates("course_number")
            .set_index("course_number")
            .to_dict(orient="index")
        )

    pool = _build_remaining_pool(
        remaining_reqs,
        dependency_lookup=dependency_lookup,
        catalog_lookup=catalog_lookup,
        prereq_df=prereq_df,
    )

    if not pool:
        return transcript_rows.reset_index(drop=True)

    completed_scheduled: set[str] = set(_completed_courses(transcript_df)) | set(
        _in_progress_courses(transcript_df)
    )

    latest_term = _latest_transcript_term(transcript_rows)
    if latest_term is None:
        next_year, next_season = _default_start_term()
    else:
        next_year, next_season = _advance_term(
            latest_term[0], latest_term[1], include_summer
        )

    scheduled_records: list[dict] = []
    terms_advanced = 0

    while pool and terms_advanced < max_future_terms:
        term_str = f"{next_year}-{next_season}"
        term_label = f"{next_season.title()} {next_year}"
        term_order = _term_order_value(next_year, next_season)
        is_summer_term = next_season == "SUMMER"
        term_credit_cap = (
            min(target_credits_per_term, 6) if is_summer_term else target_credits_per_term
        )

        eligible = [
            item
            for item in pool
            if item["prereqs"].issubset(completed_scheduled)
        ]

        if not eligible:
            next_year, next_season = _advance_term(next_year, next_season, include_summer)
            terms_advanced += 1
            continue

        eligible.sort(
            key=lambda item: (
                item["recommended_semester"],
                _CATEGORY_RANK.get(item["category"], 99),
                item["course_number"],
            )
        )

        placed_this_term: list[dict] = []
        placed_courses: set[str] = set()
        used_credits = 0.0
        lab_selected = False

        for item in eligible:
            if item["course_number"] in placed_courses:
                continue

            coreq_items: list[dict] = []
            total_credits = item["credits"]
            blocked = False
            for coreq_course in item["coreqs"]:
                if coreq_course in completed_scheduled or coreq_course in placed_courses:
                    continue
                coreq_item = next(
                    (
                        candidate
                        for candidate in pool
                        if candidate["course_number"] == coreq_course
                        and candidate["prereqs"].issubset(completed_scheduled)
                    ),
                    None,
                )
                if coreq_item is None:
                    blocked = True
                    break
                coreq_items.append(coreq_item)
                total_credits += coreq_item["credits"]

            if blocked:
                continue
            if used_credits + total_credits > term_credit_cap:
                continue

            is_lab = item["course_number"].endswith("L")
            bundle_has_lab = is_lab or any(
                c["course_number"].endswith("L") for c in coreq_items
            )
            if lab_selected and bundle_has_lab and not coreq_items:
                continue

            placed_this_term.append(item)
            placed_courses.add(item["course_number"])
            used_credits += item["credits"]
            if is_lab:
                lab_selected = True

            for coreq_item in coreq_items:
                placed_this_term.append(coreq_item)
                placed_courses.add(coreq_item["course_number"])
                used_credits += coreq_item["credits"]
                if coreq_item["course_number"].endswith("L"):
                    lab_selected = True

        for item in placed_this_term:
            scheduled_records.append(
                {
                    "node_id": f"{term_str}::{item['course_number']}::suggested",
                    "course_number": item["course_number"],
                    "course_title": item["course_title"],
                    "term": term_str,
                    "term_label": term_label,
                    "term_order": term_order,
                    "status": "suggested",
                    "credit_hours": item["credits"],
                    "grade": None,
                    "source_type": "Planned",
                    "category": item["category"],
                    "requirement_id": item["requirement_id"],
                    "requirement_name": item["requirement_name"],
                    "is_elective_slot": item["is_elective_slot"],
                    "allowed_courses": item["allowed_courses"],
                }
            )
            completed_scheduled.add(item["course_number"])
            pool.remove(item)

        next_year, next_season = _advance_term(next_year, next_season, include_summer)
        terms_advanced += 1

    unscheduled_records: list[dict] = []
    for item in pool:
        unscheduled_records.append(
            {
                "node_id": f"UNSCHEDULED::{item['course_number']}",
                "course_number": item["course_number"],
                "course_title": item["course_title"],
                "term": "UNSCHEDULED",
                "term_label": "Unscheduled",
                "term_order": 10**7,
                "status": "unscheduled",
                "credit_hours": item["credits"],
                "grade": None,
                "source_type": "Unplaced",
                "category": item["category"],
                "requirement_id": item["requirement_id"],
                "requirement_name": item["requirement_name"],
                "is_elective_slot": item["is_elective_slot"],
                "allowed_courses": item["allowed_courses"],
            }
        )

    frames = [transcript_rows]
    if scheduled_records:
        frames.append(pd.DataFrame(scheduled_records, columns=_ROADMAP_COLUMNS))
    if unscheduled_records:
        frames.append(pd.DataFrame(unscheduled_records, columns=_ROADMAP_COLUMNS))

    roadmap_df = pd.concat(frames, ignore_index=True)
    roadmap_df["_category_rank"] = (
        roadmap_df["category"].map(_CATEGORY_RANK).fillna(99)
    )
    roadmap_df = (
        roadmap_df.sort_values(["term_order", "_category_rank", "course_number"])
        .drop(columns="_category_rank")
        .reset_index(drop=True)
    )
    return roadmap_df
