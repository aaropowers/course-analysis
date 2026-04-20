from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .utils import ROOT


DB_PATH = ROOT / "app_data.db"


def init_db(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                recommended_course TEXT,
                score REAL,
                explanation TEXT
            )
            """
        )


def log_recommendations(student_id: str, recommendations: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if recommendations.empty:
        return
    with sqlite3.connect(db_path) as conn:
        recommendations.assign(student_id=student_id)[["student_id", "course_number", "score", "explanation"]].rename(
            columns={"course_number": "recommended_course"}
        ).to_sql("recommendation_runs", conn, if_exists="append", index=False)
