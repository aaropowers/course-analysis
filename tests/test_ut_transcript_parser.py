"""Smoke-test for the UT Academic Summary PDF parser.

Run directly:

    python tests/test_ut_transcript_parser.py [path/to/transcript.pdf]

If no path is provided the script looks for a default transcript sample in
Cursor's workspace PDF cache (the file the user originally attached).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _default_sample_pdf() -> Path:
    return Path(
        r"C:\Users\marvi\AppData\Roaming\Cursor\User\workspaceStorage"
        r"\b435f48bbe834b8f55c681cd29e2e955\pdfs"
        r"\925f77ae-f12e-4a41-bfd0-798b130c9b97"
        r"\University_of_Texas_Academic_Summary.pdf"
    )


def main() -> int:
    # Make `src` importable when running this file directly from repo root.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.transcript_pdf import (
        export_transcript_csv,
        parse_ut_transcript_pdf,
        to_transcript_dataframe,
    )
    from src.cleaning import parse_transcript
    from src.utils import load_catalog

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_sample_pdf()
    if not pdf_path.exists():
        print(f"[error] PDF not found: {pdf_path}")
        return 2

    print(f"[info] Parsing: {pdf_path}")
    result = parse_ut_transcript_pdf(pdf_path)

    def safe_text(value: object) -> str:
        text = str(value)
        return text.encode("cp1252", errors="replace").decode("cp1252")

    print("\n=== Metadata ===")
    print(f"  EID:            {result.metadata.eid}")
    print(f"  Name:           {result.metadata.name}")
    print(f"  Major:          {result.metadata.major}")
    print(f"  First semester: {result.metadata.first_semester}")
    print(f"  Last semester:  {result.metadata.last_semester}")
    print(f"  Classification: {result.metadata.classification}")

    print(f"\n=== Parsed {len(result.rows)} course rows ===")
    header = f"{'TERM':<13} {'RAW':<12} {'NORM':<10} {'GR':<4} {'CR':>4} {'STATUS':<14} TITLE"
    print(header)
    print("-" * len(header))
    for row in result.rows:
        credit = f"{row.credit_hours:>4.1f}" if row.credit_hours is not None else "   -"
        grade = row.grade or "-"
        print(
            f"{row.term:<13} {row.course_number_raw:<12} {row.course_number:<10} "
            f"{grade:<4} {credit} {row.status:<14} {row.title}"
        )

    if result.warnings:
        print(f"\n=== Warnings ({len(result.warnings)}) ===")
        for w in result.warnings:
            print(f"  - {safe_text(w)}")
    else:
        print("\n(no warnings)")

    # Compose an app-compatible DataFrame and run it through the existing
    # parse_transcript validation to see which courses match the catalog.
    df = to_transcript_dataframe(result)
    validated, invalid = parse_transcript(df[["course_number", "grade", "term"]].copy())

    stem = pdf_path.stem.replace(" ", "_")
    output_dir = project_root / "data" / "parsed_transcripts"
    full_csv = export_transcript_csv(result, output_dir / f"{stem}_parsed_full.csv")
    model_csv = output_dir / f"{stem}_model_input.csv"
    validated.to_csv(model_csv, index=False)
    print("\n=== CSV exports ===")
    print(f"  full parsed rows:  {full_csv}")
    print(f"  model input rows:  {model_csv}")

    catalog_codes = set(load_catalog()["course_number"])
    all_courses = df["course_number"].tolist()
    matched = [c for c in all_courses if c in catalog_codes]

    print("\n=== Catalog match summary ===")
    print(f"  total parsed rows:       {len(all_courses)}")
    print(f"  catalog-matched rows:    {len(matched)}")
    print(f"  validated (dedup) rows:  {len(validated)}")
    print(f"  unknown-to-catalog rows: {len(invalid)}")

    if invalid:
        print("\nCourses not in catalog (expected for non-ME / transfer electives):")
        for code in invalid:
            print(f"  - {code}")

    print("\nSample of validated rows (head):")
    print(validated.head(15).to_string(index=False))

    # Minimal acceptance checks based on transcript coverage.
    # Spring_2025 snapshot does not include later Spring 2026 courses.
    if "Spring_2025_Transcript" in pdf_path.stem.replace(" ", "_"):
        expected_minimums = {
            "ME318M",
            "ME330",
            "ME335",
            "ME339",
        }
    else:
        expected_minimums = {
            "ME318M",  # PROGRAM & ENGR COMP METHODS
            "ME330",   # FLUID MECHANICS
            "ME335",   # ENGINEERING STATISTICS
            "ME339",   # HEAT TRANSFER (B+ grade)
            "ME344",   # DYNAMIC SYSTEMS AND CONTROLS
            "ME365D",  # DATA SCIENCE FOR ENGINEERS (in-progress, no grade)
            "M427L",   # ADV CALCULUS FOR APPLICATNS II (in-progress)
        }
    missing = expected_minimums - set(df["course_number"])
    if missing:
        print(f"\n[FAIL] Missing expected course codes: {sorted(missing)}")
        return 1

    # Verify ME339 row exists.
    me339_rows = df[df["course_number"] == "ME339"]
    if me339_rows.empty:
        print("[FAIL] Expected ME339 row to be present")
        return 1

    if "Spring_2025_Transcript" not in pdf_path.stem.replace(" ", "_"):
        # Verify ME365D is marked in-progress (no grade).
        me365d = df[df["course_number"] == "ME365D"].iloc[0]
        if me365d["status"] != "in_progress":
            print(f"[FAIL] Expected ME365D status in_progress, got {me365d['status']!r}")
            return 1

    # Verify a transfer row is classified correctly.
    his_rows = df[df["course_number"] == "HIS315K"]
    if his_rows.empty or his_rows.iloc[0]["status"] != "transfer":
        print("[FAIL] Expected HIS315K to parse with status 'transfer'")
        return 1

    # Verify a credit-by-exam row is classified correctly.
    phy_rows = df[df["course_number"] == "PHY303K"]
    if phy_rows.empty or phy_rows.iloc[0]["status"] != "credit_by_exam":
        print("[FAIL] Expected PHY303K to parse with status 'credit_by_exam'")
        return 1

    print("\n[PASS] UT transcript parser smoke test succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
