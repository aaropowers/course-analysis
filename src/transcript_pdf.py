"""Deterministic parser for UT Austin Academic Summary unofficial transcripts.

This module extracts course-level records (course number, title, grade, term,
credit hours, status) from the PDF text produced by `pypdf`. The parser is
built around the fixed column-per-line layout emitted by UT's Academic
Summary PDF and is intentionally deterministic so behavior is easy to reason
about and debug.

Primary entry points:

- `parse_ut_transcript_pdf(source)` -- accepts a file path, bytes, or a
  file-like object (e.g. a Streamlit upload) and returns a structured
  `TranscriptParseResult`.
- `to_transcript_dataframe(result)` -- converts the parsed rows into a
  `pandas.DataFrame` compatible with the existing `transcript_df` schema
  (`course_number`, `grade`, `term`, `grade_points`) plus a few extra
  columns (`title`, `credit_hours`, `status`, `source_type`) that downstream
  code can treat as additive.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import io
import re
from typing import IO, Iterable, Union

import pandas as pd

from .utils import GRADE_POINTS, normalize_course_number


PdfSource = Union[str, Path, bytes, bytearray, IO[bytes]]


# ---------------------------------------------------------------------------
# Regex/line classification
# ---------------------------------------------------------------------------

TERM_HEADER_RE = re.compile(r"^(Fall|Spring|Summer)\s+(\d{4})\s+Courses$")

# Matches UT-style course numbers on a single extracted line.
# Examples: "HIS 315K", "M 408C", "C C 303", "M E 318M", "SDS 322E",
# "E S 333T", "FIN 322F", "UGS 016".
COURSE_NUMBER_RE = re.compile(r"^[A-Z]{1,4}(?:\s[A-Z]{1,2}){0,2}\s\d{3}[A-Z]?$")
COURSE_START_RE = re.compile(r"^(?P<course>[A-Z]{1,4}(?:\s[A-Z]{1,2}){0,2}\s\d{3}[A-Z]?)(?:\s+(?P<rest>.*))?$")

LETTER_GRADE_RE = re.compile(r"^(A|A-|B\+|B|B-|C\+|C|C-|D\+|D|D-|F|CR|IP|Q|W)$")
INTEGER_RE = re.compile(r"^\d+$")
DECIMAL_RE = re.compile(r"^\d+\.\d+$")

_COLUMN_HEADERS = {
    "Course",
    "Title",
    "Grade",
    "Unique",
    "Type",
    "Credit Hours",
    "Grade Points",
}

_META_PREFIXES = (
    "EID:",
    "Name:",
    "School ",
    "Major ",
    "First Semester Enrolled:",
    "Last Semester Enrolled:",
    "Date Degree Expected:",
    "Classification:",
)

_BOILERPLATE_EXACT = {
    "Academic Summary Unofficial Document",
    "Academic Summary",
    "Unofficial Document",
    "The University of Texas at Austin",
    "HONORS",
}

_PAGE_RE = re.compile(r"^Page \d+ of \d+$")

_TOTALS_PREFIXES = (
    "Total Hours Transferred:",
    "Total Hours Taken:",
    "GPA Hours:",
    "Grade Points:",
    "GPA:",
    "Hours:",
)

_TOTALS_EXACT = {"Lower Division", "Graduate Level", "Upper Division", "Overall"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParsedCourseRow:
    """A single course row extracted from the transcript."""

    course_number_raw: str
    course_number: str
    title: str
    grade: str | None
    unique: str | None
    course_type: str
    credit_hours: float | None
    grade_points: float | None
    term_season: str
    term_year: int
    term: str
    status: str  # completed | in_progress | transfer | credit_by_exam


@dataclass
class TranscriptMetadata:
    eid: str | None = None
    name: str | None = None
    major: str | None = None
    first_semester: str | None = None
    last_semester: str | None = None
    classification: str | None = None


@dataclass
class TranscriptParseResult:
    rows: list[ParsedCourseRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: TranscriptMetadata = field(default_factory=TranscriptMetadata)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(source: PdfSource) -> str:
    """Return all text from a PDF source using pypdf, joined with newlines.

    Accepts a filesystem path, raw bytes, or a binary file-like object such
    as the object returned by `streamlit.file_uploader`.
    """
    from pypdf import PdfReader

    if isinstance(source, (bytes, bytearray)):
        reader = PdfReader(io.BytesIO(bytes(source)))
    elif isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
    else:
        # Assume file-like; rewind if possible.
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        reader = PdfReader(source)

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Token stream parsing
# ---------------------------------------------------------------------------

def _is_boilerplate(line: str) -> bool:
    if line.startswith("Academic Summary Unofficial Document"):
        return True
    if line.startswith("Academic Summary Uno"):
        return True
    if line.startswith("The University of T"):
        return True
    if line == "•":
        return True
    if line in _BOILERPLATE_EXACT or line in _COLUMN_HEADERS:
        return True
    if "Course Title Grade Unique Type Credit Hours Grade Points" in line:
        return True
    if _PAGE_RE.match(line):
        return True
    if line.startswith(_META_PREFIXES):
        return True
    return False


def _is_totals_line(line: str) -> bool:
    if line in _TOTALS_EXACT:
        return True
    return line.startswith(_TOTALS_PREFIXES)


def _clean_lines(raw_text: str) -> list[str]:
    """Strip whitespace, split tab-separated chunks, and drop empty lines."""
    cleaned: list[str] = []
    for ln in raw_text.splitlines():
        # Some transcript exports use tab-separated columns on a single line.
        # Split tabs so the downstream parser sees a consistent token stream.
        tab_chunks = [chunk.strip() for chunk in ln.split("\t")]
        for chunk in tab_chunks:
            if not chunk:
                continue
            cleaned.append(chunk)
    return _merge_split_course_tokens(cleaned)


def _is_department_token(token: str) -> bool:
    return bool(re.match(r"^[A-Z]{1,4}(?:\s[A-Z]{1,2}){0,2}$", token))


def _is_number_token(token: str) -> bool:
    return bool(re.match(r"^\d{3}[A-Z]?$", token))


def _merge_split_course_tokens(tokens: list[str]) -> list[str]:
    """Merge course-number fragments split across lines.

    Example conversions:
      - "BIO", "311D" -> "BIO 311D"
      - "M E", "318M" -> "M E 318M"
      - "PHY", "303K" -> "PHY 303K"
    """
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        current = tokens[i]
        if i + 1 < len(tokens) and _is_department_token(current) and _is_number_token(tokens[i + 1]):
            merged.append(f"{current} {tokens[i + 1]}")
            i += 2
            continue
        merged.append(current)
        i += 1
    return merged


def _extract_metadata(lines: Iterable[str]) -> TranscriptMetadata:
    meta = TranscriptMetadata()
    for ln in lines:
        if ln.startswith("EID:"):
            meta.eid = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("Name:"):
            meta.name = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("Major 1:"):
            meta.major = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("First Semester Enrolled:"):
            meta.first_semester = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("Last Semester Enrolled:"):
            meta.last_semester = ln.split(":", 1)[1].strip() or None
        elif ln.startswith("Classification:"):
            meta.classification = ln.split(":", 1)[1].strip() or None
    return meta


def _find_term_blocks(lines: list[str]) -> list[tuple[int, str, int]]:
    """Return (index, season_upper, year) for each term header line."""
    positions: list[tuple[int, str, int]] = []
    for idx, ln in enumerate(lines):
        match = TERM_HEADER_RE.match(ln)
        if match:
            positions.append((idx, match.group(1).upper(), int(match.group(2))))
    return positions


def _find_transcript_end(lines: list[str]) -> int:
    for idx, ln in enumerate(lines):
        normalized = ln.replace(" ", "")
        if ln.startswith("Total Hours Transferred:") or normalized.startswith("TotalHoursTransferred:"):
            return idx
    return len(lines)


def _collect_block_tokens(lines: list[str], start_exclusive: int, end_exclusive: int) -> list[str]:
    """Pull parsable tokens between two line indices, dropping boilerplate."""
    tokens: list[str] = []
    for j in range(start_exclusive + 1, end_exclusive):
        ln = lines[j]
        if _is_boilerplate(ln) or _is_totals_line(ln):
            continue
        tokens.append(ln)
    return tokens


def _parse_course_type(tokens: list[str], i: int) -> tuple[str | None, int]:
    """Return (course_type, new_index) after consuming type tokens."""
    if i >= len(tokens):
        return None, i
    t = tokens[i]
    if t in ("Transfer", "In residence"):
        return t, i + 1
    # "Credit by" + "exam" split across two lines in the UT PDF layout.
    if t == "Credit by" and i + 1 < len(tokens) and tokens[i + 1] == "exam":
        return "Credit by exam", i + 2
    return None, i


def _classify_status(course_type: str | None, grade: str | None) -> str:
    if course_type == "Transfer":
        return "transfer"
    if course_type == "Credit by exam":
        return "credit_by_exam"
    if grade is None:
        return "in_progress"
    if grade == "IP":
        return "in_progress"
    return "completed"


def _parse_token_stream(
    tokens: list[str], season: str, year: int
) -> tuple[list[ParsedCourseRow], list[str]]:
    rows: list[ParsedCourseRow] = []
    warnings: list[str] = []
    i = 0
    while i < len(tokens):
        start_match = COURSE_START_RE.match(tokens[i])
        if not start_match:
            warnings.append(
                f"[{year}-{season}] skipped unrecognized line before course: {tokens[i]!r}"
            )
            i += 1
            continue

        course_raw = start_match.group("course")
        chunk_parts = [start_match.group("rest") or ""]
        j = i + 1
        while j < len(tokens):
            if COURSE_START_RE.match(tokens[j]):
                break
            chunk_parts.append(tokens[j])
            joined = " ".join(part for part in chunk_parts if part).strip()
            # Once credits/points are found, stop collecting this row.
            if re.search(r"\d+\.\d+\s+\d+\.\d+$", joined):
                j += 1
                break
            j += 1
        i = j

        chunk_text = " ".join(part for part in chunk_parts if part).strip()
        chunk_text = re.sub(r"\s+", " ", chunk_text)
        chunk_text = chunk_text.replace("Credit by exam", "Credit by exam")
        chunk_text = chunk_text.replace("Credit by", "Credit by")
        chunk_text = chunk_text.replace("In residence", "In residence")
        chunk_text = chunk_text.replace(" In residence ", " In residence ")

        # Parse trailing credits/points first.
        credit_hours: float | None = None
        grade_points: float | None = None
        tail_match = re.search(r"(?P<credits>\d+\.\d+)\s+(?P<points>\d+\.\d+)$", chunk_text)
        if tail_match:
            credit_hours = float(tail_match.group("credits"))
            grade_points = float(tail_match.group("points"))
            core_text = chunk_text[: tail_match.start()].strip()
        else:
            core_text = chunk_text

        course_type: str | None = None
        for ctype in ("Credit by exam", "In residence", "Transfer"):
            if core_text.endswith(ctype):
                course_type = ctype
                core_text = core_text[: -len(ctype)].strip()
                break
        if course_type is None and core_text.endswith("Credit by"):
            course_type = "Credit by exam"
            core_text = core_text[: -len("Credit by")].strip()

        unique: str | None = None
        unique_match = re.search(r"(?P<unique>\d+)$", core_text)
        if unique_match:
            unique = unique_match.group("unique")
            core_text = core_text[: unique_match.start()].strip()
        else:
            warnings.append(
                f"[{year}-{season}] could not locate unique number for {course_raw} {chunk_text!r}"
            )
            continue

        grade: str | None = None
        grade_match = re.search(r"(A-|A|B\+|B|B-|C\+|C|C-|D\+|D|D-|F|CR|IP|Q|W)$", core_text)
        if grade_match:
            grade = grade_match.group(1)
            title = core_text[: grade_match.start()].strip()
        else:
            title = core_text.strip()

        rows.append(
            ParsedCourseRow(
                course_number_raw=course_raw,
                course_number=normalize_course_number(course_raw),
                title=title,
                grade=grade,
                unique=unique,
                course_type=course_type or "",
                credit_hours=credit_hours,
                grade_points=grade_points,
                term_season=season,
                term_year=year,
                term=f"{year}-{season}",
                status=_classify_status(course_type, grade),
            )
        )

    return rows, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ut_transcript_text(text: str) -> TranscriptParseResult:
    """Parse raw transcript text into a `TranscriptParseResult`.

    Intended for testing and for callers that already have text extracted.
    Production callers should prefer `parse_ut_transcript_pdf`.
    """
    lines = _clean_lines(text)
    metadata = _extract_metadata(lines)

    term_positions = _find_term_blocks(lines)
    end_idx = _find_transcript_end(lines)

    result = TranscriptParseResult(metadata=metadata, raw_text=text)
    if not term_positions:
        result.warnings.append("No term headers (e.g. 'Fall 2024 Courses') detected.")
        return result

    boundaries = term_positions + [(end_idx, "", 0)]
    for i in range(len(boundaries) - 1):
        start_idx, season, year = boundaries[i]
        next_start_idx = boundaries[i + 1][0]
        tokens = _collect_block_tokens(lines, start_idx, next_start_idx)
        section_rows, section_warnings = _parse_token_stream(tokens, season, year)
        result.rows.extend(section_rows)
        result.warnings.extend(section_warnings)

    return result


def parse_ut_transcript_pdf(source: PdfSource) -> TranscriptParseResult:
    """Extract PDF text and parse it into a `TranscriptParseResult`."""
    text = extract_pdf_text(source)
    return parse_ut_transcript_text(text)


def to_transcript_dataframe(result: TranscriptParseResult) -> pd.DataFrame:
    """Convert parsed rows into a dataframe shaped like the app's transcript.

    Columns returned:
        course_number, grade, term, grade_points,
        title, credit_hours, status, source_type, course_number_raw, unique

    Downstream code only requires `course_number`; extra columns are
    additive metadata the UI can surface.
    """
    if not result.rows:
        return pd.DataFrame(
            columns=[
                "course_number",
                "grade",
                "term",
                "grade_points",
                "title",
                "credit_hours",
                "status",
                "source_type",
                "course_number_raw",
                "unique",
            ]
        )

    records = []
    for row in result.rows:
        record = asdict(row)
        # Align with existing grade_points convention: map letter grade through
        # the app's canonical GRADE_POINTS table. We intentionally ignore the
        # PDF's "Grade Points" column (which is credit_hours * gpa_points)
        # because downstream features treat grade_points as per-course GPA.
        gp = GRADE_POINTS.get(row.grade) if row.grade else None
        record["grade_points"] = gp
        record["source_type"] = row.course_type
        records.append(record)

    df = pd.DataFrame.from_records(records)
    # Keep a stable column order; downstream code should be tolerant of extras.
    preferred = [
        "course_number",
        "grade",
        "term",
        "grade_points",
        "title",
        "credit_hours",
        "status",
        "source_type",
        "course_number_raw",
        "unique",
    ]
    remaining = [c for c in df.columns if c not in preferred]
    return df[preferred + remaining]


def export_transcript_csv(
    result: TranscriptParseResult,
    output_path: str | Path,
    *,
    validated_only: bool = False,
) -> Path:
    """Write parsed transcript rows to CSV and return output path.

    Args:
        result: Parsed transcript output.
        output_path: Destination CSV path.
        validated_only: If True, write only the model-facing columns commonly
            consumed by the app (`course_number`, `grade`, `term`,
            `grade_points`). If False, include all parsed metadata columns.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = to_transcript_dataframe(result)
    if validated_only:
        keep_cols = ["course_number", "grade", "term", "grade_points"]
        df = df[[column for column in keep_cols if column in df.columns]]

    df.to_csv(out_path, index=False)
    return out_path
