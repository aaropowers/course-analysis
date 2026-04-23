from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.db import init_db
from src.utils import load_transcripts


st.set_page_config(
    page_title="ME Course Planner",
    page_icon="⚙️",
    layout="wide",
)

init_db()

if "transcript_df" not in st.session_state:
    demo_transcripts = load_transcripts()
    st.session_state["transcript_df"] = demo_transcripts[demo_transcripts["student_id"] == "demo_mid"].copy()
    st.session_state["active_student_id"] = "demo_mid"
    st.session_state["selected_interests"] = ["controls", "robotics"]
    st.session_state["target_credit_load"] = 12
    st.session_state["invalid_courses"] = []

st.title("Transcript-Aware ME Course Planner")


st.subheader("Why did we make this app?")
project_root = Path(__file__).resolve().parent
intro_image_candidates = [
    project_root / "assets" / "why_we_made_this_app.png",
    project_root / "assets" / "why_we_made_this_app.jpg",
    project_root / "assets" / "why_we_made_this_app.jpeg",
]
intro_image_path = next((path for path in intro_image_candidates if path.exists()), None)
if intro_image_path is not None:
    st.image(
        str(intro_image_path),
        caption="Planning classes should feel clear, not overwhelming.",
        use_container_width=True,
    )
else:
    st.info(
        "Add an image at `assets/why_we_made_this_app.png` "
        "(or `.jpg` / `.jpeg`) to display it in this section."
    )
st.markdown(
    """
    Degree planning can be overwhelming, especially in programs like Mechanical Engineering where
    prerequisites, corequisites, and timing constraints are tightly connected across semesters.

    We built this app to make that process easier and more transparent by turning transcript data
    into a clear roadmap of what you have completed, what is in progress, and what to take next.

    Ninety percent of entering freshmen think they’ll graduate within four years, yet only 45 percent of them will.
    Better visibility into degree progress and course sequencing can help students make earlier,
    better-informed decisions.
    """
)
st.caption(
    "Source: The Hechinger Report "
    "(https://hechingerreport.org/how-the-college-lobby-got-the-government-to-measure-graduation-rates-over-six-years-instead-of-four/)"
)

st.subheader("How to use this app")
st.markdown(
    """
    Use the pages in the sidebar to:

    1. load a transcript manually or from CSV
    2. review degree progress and locked requirements
    3. inspect explainable course recommendations
    4. build a balanced next-semester bundle
    """
)

st.info(
    "This demo uses a curated mechanical engineering requirement map and synthetic student records "
    "for recommender behavior. It is intended for prototype demonstrations, not official advising."
)
