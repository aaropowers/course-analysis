# Mechanical Engineering Course Planner

Transcript-aware Streamlit prototype for planning UT Mechanical Engineering coursework.

The app helps a student:

- load a transcript (demo profile, CSV, manual course picker, or UT Academic Summary PDF)
- audit degree progress with requirement status detail
- view explainable course recommendations with predicted GPA ranges
- generate a graduation roadmap that combines past terms with suggested future terms
- inspect coursework-backed analytics from a bootstrapped dataset

This is a demo-oriented prototype, not an official advising system.

## Current App Pages

The app is organized into five pages:

1. `Input`
   - Load a demo transcript, upload transcript CSV, or parse a UT-format unofficial transcript PDF.
   - PDF parsing is cached in session state so users can navigate away and come back without re-uploading.
   - Export available: `Export CSV (full parsed output)`.
   - Set interest areas and target credit load used by downstream pages.
2. `Degree Audit`
   - Audits each requirement in `degree_plan.csv` against the active transcript.
   - Tracks statuses including `Completed`, `In Progress`, `Eligible`, `Eligible with corequisite`, and `Locked`.
   - Includes both a degree-requirement map and a transcript timeline map.
3. `Recommendations`
   - Ranks eligible courses with explainable signal breakdowns.
   - Shows predicted GPA ranges and confidence.
   - Uses recommendation logging via SQLite.
4. `Semester Planner`
   - Builds a graduation roadmap from current transcript state to completion.
   - Supports future-term controls: max credits, optional summer terms, and planning horizon.
   - Shows unscheduled requirements when constraints prevent placement.
5. `Insights`
   - Analytics views from `coursework_bootstrapped.csv`:
     grade distributions, co-enrollment effects, department GPA heatmap, and top same-term pairs.

## High-Level Methodology

The project combines four layers:

- `Structured degree rules`
  - `degree_plan.csv` + `prereqs.csv` drive audit status and scheduling constraints.
- `Transcript normalization and parsing`
  - CSV/manual inputs are normalized through `src.cleaning`.
  - UT unofficial transcript PDFs are parsed with `pypdf` via `src.transcript_pdf`.
- `Recommendation scoring`
  - `src.recommender` ranks courses using requirement priority, timing fit, similarity, co-enrollment, collaborative signals, and interests.
- `Roadmap scheduling`
  - `src.semester_planner.build_graduation_roadmap` assigns unmet requirements to future terms with prerequisite/corequisite awareness, credit caps, and unscheduled fallback.

## Data Pipeline

Coursework data is maintained in stages:

- `coursework.csv`
  Original raw dataset (kept untouched).
- `data/coursework_cleaned.csv`
  Canonicalized and validated version of raw records.
- `data/coursework_bootstrapped.csv`
  Expanded dataset for robust recommendation evidence and insights.

Rebuild cleaned/bootstrapped coursework with:

```bash
python generate_coursework_dataset.py
```

## Main Data Files

- `data/course_catalog.csv`
  Canonical course metadata (title, description, credits, tags, recommended semester, etc.).
- `data/degree_plan.csv`
  Curated ME requirement map used by audit and roadmap logic.
- `data/prereqs.csv`
  Prerequisite/corequisite graph.
- `data/elective_groups.csv`
  Elective and interest-area groupings.
- `data/synthetic_transcripts.csv`
  Demo transcript profiles loaded on app start.
- `data/coursework_bootstrapped.csv`
  Expanded coursework records for analytics and evidence.
- `data/parsed_transcripts/`
  Example parsed transcript outputs generated during testing/demo.

## Core Modules

- `app.py`
  Streamlit entry page and app-level session bootstrapping.
- `src/transcript_pdf.py`
  UT Academic Summary PDF parsing and dataframe conversion.
- `src/cleaning.py`
  Transcript normalization for CSV/manual inputs.
- `src/audit.py`
  Degree audit logic, requirement status classification, and progression builders.
- `src/recommender.py`
  Explainable recommendation scoring and prediction fields.
- `src/semester_planner.py`
  Graduation roadmap scheduler and semester planning utilities.
- `src/plot_utils.py`
  Shared timeline flowchart rendering for Degree Audit and Semester Planner.
- `src/features.py`
  Coursework feature engineering and analytics helpers.
- `src/coursework_bootstrap.py`
  Raw-to-cleaned-to-bootstrapped dataset pipeline.
- `src/utils.py`
  Shared loaders, normalization helpers, GPA/hours utilities, and cache helpers.
- `src/db.py`
  SQLite initialization and recommendation logging.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Repository Layout

```text
course-analysis/
├── app.py
├── README.md
├── requirements.txt
├── app_data.db
├── coursework.csv
├── generate_coursework_dataset.py
├── data/
│   ├── course_catalog.csv
│   ├── coursework_cleaned.csv
│   ├── coursework_bootstrapped.csv
│   ├── degree_plan.csv
│   ├── prereqs.csv
│   ├── elective_groups.csv
│   ├── synthetic_transcripts.csv
│   ├── demo_upload_template.csv
│   └── parsed_transcripts/
├── pages/
│   ├── 1_Input.py
│   ├── 2_Degree_Audit.py
│   ├── 3_Recommendations.py
│   ├── 4_Semester_Planner.py
│   └── 5_Insights.py
└── src/
    ├── audit.py
    ├── cleaning.py
    ├── coursework_bootstrap.py
    ├── db.py
    ├── features.py
    ├── plot_utils.py
    ├── recommender.py
    ├── semester_planner.py
    ├── transcript_pdf.py
    ├── utils.py
    └── __init__.py
```

## Important Notes

- This app is a planning assistant, not official advising.
- Degree rules and prerequisite mappings are curated and may simplify registrar/advising edge cases.
- Recommendation and roadmap outputs are explainable but heuristic; they are not guaranteed-optimal.
- Bootstrapped coursework data supports testing and demo analytics, but it is not institutional ground truth.
