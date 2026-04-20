# Mechanical Engineering Course Planner

A lightweight Streamlit prototype that helps a mechanical engineering student understand degree progress, identify remaining requirements, and get explainable recommendations for future coursework.

This project is designed as a one-week team prototype. The goal is to deliver a reliable, polished demo rather than a production advising platform.

## Overview

The app combines structured degree rules with notebook-inspired recommendation logic to support four core student questions:

- What requirements have I already completed?
- What is still missing from my degree plan?
- Which courses am I eligible to take next?
- What courses make sense for my next semester?

## MVP Features

- Manual transcript entry
- CSV transcript upload
- Degree audit against a curated ME degree plan
- Remaining requirement and locked-course detection
- Ranked next-course recommendations
- Plain-language recommendation explanations
- Suggested semester bundle based on credit load and recommendation strength
- Basic supporting insights and charts for demo storytelling

## Tech Stack

- `Streamlit` for the web app
- `Python` for backend logic
- `pandas` and `numpy` for data handling
- `scikit-learn` for TF-IDF similarity and collaborative-style recommendation support
- `plotly` for charts
- `SQLite` for lightweight logging of recommendation runs
- `CSV` files for prototype reference data

## Project Structure

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
├── src/
│   ├── cleaning.py
│   ├── audit.py
│   ├── coursework_bootstrap.py
│   ├── recommender.py
│   ├── features.py
│   ├── semester_planner.py
│   ├── db.py
│   └── utils.py
└── notebooks / source files
    ├── course_analysis.ipynb
    ├── coursework_data.xlsx
    ├── fall_2026_me_gateway_electives.xlsx
    └── me-general-curriculum_26-28.pdf
```

## How the App Works

### 1. Input

The user can:

- load a prepared demo profile
- upload a transcript CSV
- manually select completed courses
- choose interest areas such as controls, robotics, thermal, manufacturing, data, materials, design, or energy
- set a target credit load for the next term

### 2. Degree Audit

The app compares the student's completed courses against a curated degree plan and shows:

- completed requirements
- remaining requirements
- degree progress percentage
- completed required hours
- locked courses with missing prerequisites or corequisites

### 3. Recommendations

The recommendation engine ranks eligible future courses using a hybrid score built from:

- remaining requirement priority
- recommended-semester fit
- TF-IDF similarity between course descriptions
- co-enrollment patterns from transcript histories
- lightweight collaborative filtering from historical student-course patterns
- student interest alignment

Each recommendation includes a plain-language explanation so the ranking is easy to defend during a demo.

### 4. Semester Planner

The semester planner builds a suggested bundle of courses based on:

- recommendation strength
- remaining core requirements
- interest match
- target credit load
- simple handling for prerequisite and corequisite compatibility

### 5. Insights

The insights page provides supporting visuals such as:

- top co-enrollment pairs in the demo dataset
- similar courses based on description similarity

## Data Files

### `data/course_catalog.csv`

Prototype course catalog with:

- course number
- title
- description
- credits
- average GPA proxy
- elective category
- interest tags
- recommended semester

This file is also the canonical metadata source for coursework dataset cleaning and bootstrapping. When a course appears in the catalog, the cleaned and bootstrapped coursework outputs always use this catalog-backed title and description.

### `data/degree_plan.csv`

Curated degree requirement map used for the audit engine. This is the main source of requirement logic in the prototype.

### `data/prereqs.csv`

Simplified prerequisite and corequisite relationships used to determine eligibility and locked courses.

### `data/elective_groups.csv`

Grouping file for gateway electives and interest-area tagging.

### `data/synthetic_transcripts.csv`

Small synthetic student histories used to support recommendation behavior and demo scenarios in the current Streamlit experience.

### `coursework.csv`

The original raw notebook dataset. This file is intentionally preserved as the unclean source of truth and may contain formatting inconsistencies, duplicate description variants, and noisy grades or semester labels.

### `data/coursework_cleaned.csv`

Canonicalized version of the raw coursework file. It normalizes:

- course numbers
- semesters
- grades
- institutions
- departments
- course descriptions

It also adds:

- `department`
- `course_title`
- `record_source`

### `data/coursework_bootstrapped.csv`

Expanded testing dataset built from the cleaned source. It preserves the base notebook-compatible columns while adding derived metadata and synthetic rows for larger-scale testing.

## Coursework Dataset Generation

To rebuild the cleaned and expanded coursework files from the raw notebook source:

```bash
python generate_coursework_dataset.py
```

This process:

- cleans the raw `coursework.csv`
- normalizes course numbers, semesters, grades, and institutions
- derives `department`
- forces canonical `course_title` and `course_description` from `data/course_catalog.csv` where available
- validates tracked ME prerequisite/corequisite sequencing on candidate student templates
- bootstraps additional student trajectories from valid cleaned templates
- writes `data/coursework_cleaned.csv` and `data/coursework_bootstrapped.csv`

Utilities in `src/utils.py` can now load these files centrally through:

```python
from src.utils import load_coursework_records

raw_df = load_coursework_records("raw")
cleaned_df = load_coursework_records("cleaned")
boot_df = load_coursework_records("bootstrapped")
```

## Demo Profiles

The app includes three prepared student scenarios:

- `demo_early`: mostly foundational math and science completed
- `demo_mid`: progressing through the ME core sequence
- `demo_upper`: ready for advanced controls, robotics, data, and gateway electives

These profiles are meant to produce visibly different audits and recommendations during a live demo.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Demo flow

Recommended order during a presentation:

1. `Input`
2. `Degree Audit`
3. `Recommendations`
4. `Semester Planner`
5. `Insights`

## Core Modules

### `src/cleaning.py`

Handles transcript normalization and validation for the app workflow.

### `src/audit.py`

Computes degree progress, missing requirements, eligible courses, and locked courses.

### `src/features.py`

Builds reusable recommendation features such as course similarity, co-enrollment counts, and the student-course matrix.

### `src/recommender.py`

Combines feature signals into an explainable hybrid recommendation score.

### `src/semester_planner.py`

Builds a next-semester bundle from ranked recommendations and credit constraints.

### `src/db.py`

Initializes the SQLite database and stores recommendation runs.

### `src/coursework_bootstrap.py`

Builds the cleaned and bootstrapped coursework datasets from the raw notebook source, enforces canonical catalog descriptions, derives departments, validates tracked prerequisite/corequisite sequencing, and writes output CSV assets.

## Prototype Assumptions

This version intentionally simplifies several things:

- The degree plan is curated for the prototype and is not a certified advising source.
- Prerequisite logic is simplified to support a reliable demo.
- Recommendation behavior depends partly on synthetic transcript data.
- The semester planner suggests a reasonable bundle, not an optimized or guaranteed graduation path.
- The expanded coursework dataset is a testing asset built from a small sample and preserves broad observed behavior rather than institutional-scale truth.

## Out of Scope for This Version

- transcript PDF parsing or OCR
- authentication and user accounts
- live integration with university systems
- schedule/time-table optimization
- multi-major support
- open-ended chatbot advising
- graduation guarantees

## Team Handoff Notes

- Data and degree logic updates should primarily go into `data/degree_plan.csv`, `data/prereqs.csv`, and `data/course_catalog.csv`.
- Recommendation tuning should happen in `src/recommender.py` and `src/features.py`.
- Streamlit interface updates belong in `app.py` and `pages/`.
- Demo scenarios can be extended in `data/synthetic_transcripts.csv`.
- The larger notebook testing datasets are regenerated through `generate_coursework_dataset.py`, not by editing the generated CSVs manually.

## Suggested Future Improvements

- wire the expanded coursework dataset into dashboard histogram features
- expose a dataset selector in the app for demo vs coursework-backed analysis
- add OR-condition and minimum-grade prerequisite logic
- add stronger validation reports for template-student filtering
- support additional canonical catalog coverage for non-ME electives
- deploy publicly through Streamlit Community Cloud or Render

## Disclaimer

This application is a transcript-aware course planning prototype for demonstration purposes. It should be presented as a planning assistant, not as an official academic advising system.
