# Mechanical Engineering Course Planner

Streamlit prototype for transcript-aware mechanical engineering course planning. The app helps a student upload or enter completed coursework, audit progress against a curated ME degree plan, view explainable next-course recommendations, build a suggested semester bundle, and explore coursework-backed analytics.

This is a demo-oriented prototype, not an official advising system.

## What the App Does

- accepts completed coursework by manual entry or CSV upload
- audits progress against the ME degree plan
- identifies remaining requirements and locked courses
- recommends eligible next courses with short explanations
- suggests a next-semester bundle based on credit load and course fit
- shows insights from a larger bootstrapped coursework dataset

## App Flow

The Streamlit app is organized into five pages:

1. `Input`
   Collects completed courses, interests, and target credit load.
2. `Degree Audit`
   Maps completed work to the degree plan and shows what is done, remaining, and still locked.
3. `Recommendations`
   Ranks eligible courses and shows evidence-backed recommendation cards.
4. `Semester Planner`
   Builds a compact suggested bundle while respecting credit load and corequisite handling.
5. `Insights`
   Shows coursework-backed analytics such as grade distributions, co-enrollment effects, department GPA heatmaps, and common same-term course pairs.

## High-Level Methodology

The project combines three layers:

- `Structured degree rules`
  `degree_plan.csv` and `prereqs.csv` drive the audit, eligibility checks, and requirement tracking.
- `Hybrid recommendation scoring`
  Recommendations are ranked with a mix of requirement priority, semester fit, text similarity, co-enrollment patterns, collaborative-style support, and interest alignment.
- `Coursework-backed evidence`
  A cleaned and bootstrapped coursework dataset is used for analytics and compact evidence signals such as average GPA, pass rate, common companions, and historical sequencing support.

The ranking model is still lightweight and explainable by design. The goal is a reliable prototype, not a production advising engine.

## Data Pipeline

The coursework data has three stages:

- `coursework.csv`
  Original raw notebook dataset. This file is kept untouched.
- `data/coursework_cleaned.csv`
  Cleaned and canonicalized version of the raw dataset.
- `data/coursework_bootstrapped.csv`
  Expanded testing dataset used for analytics and evidence-backed insights.

The cleaning and bootstrapping pipeline:

- normalizes course numbers, semesters, grades, and institutions
- derives `department`
- forces canonical course titles and descriptions from `data/course_catalog.csv` where available
- validates tracked prerequisite/corequisite order in student templates
- bootstraps additional synthetic student histories
- adds plausible fallback grade distributions for missing degree-plan courses with no observed raw grade history

Rebuild the cleaned and bootstrapped coursework files with:

```bash
python generate_coursework_dataset.py
```

## Main Data Files

- `data/course_catalog.csv`
  Canonical course metadata: title, description, department, credits, average GPA proxy, tags, and recommended semester.
- `data/degree_plan.csv`
  Curated ME requirement map used by the audit engine.
- `data/prereqs.csv`
  Simplified prerequisite and corequisite relationships.
- `data/elective_groups.csv`
  Elective and interest-area grouping support.
- `data/synthetic_transcripts.csv`
  Small demo transcript set used by the app for demo profiles and part of the ranking signals.
- `data/coursework_bootstrapped.csv`
  Larger testing dataset used for analytics and recommendation evidence.

## Core Modules

- `app.py`
  Streamlit entry point.
- `src/cleaning.py`
  Transcript normalization for app inputs.
- `src/audit.py`
  Degree audit, remaining requirements, and eligibility logic.
- `src/recommender.py`
  Hybrid recommendation scoring plus evidence-backed explanations.
- `src/semester_planner.py`
  Bundle builder for the next semester.
- `src/features.py`
  Shared feature engineering and coursework analytics helpers.
- `src/coursework_bootstrap.py`
  Raw-to-cleaned-to-bootstrapped coursework generation pipeline.
- `src/utils.py`
  Central data loaders and normalization helpers.
- `src/db.py`
  SQLite logging for recommendation runs.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Repository Layout

```text
course analysis/
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
│   └── demo_upload_template.csv
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
    ├── recommender.py
    ├── semester_planner.py
    ├── utils.py
    └── __init__.py
```

## Important Notes

- The degree map and prerequisite logic are curated for the prototype and may simplify official advising rules.
- Recommendation outputs are explainable but not guaranteed-optimal.
- The bootstrapped coursework dataset is useful for testing and demo analytics, but it is not institutional ground truth.
- The app should be framed as a planning assistant, not a certified advising tool.
