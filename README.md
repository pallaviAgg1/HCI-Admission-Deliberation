# Admissions Reflection Prototype

Single source of truth for setup, current flow, scoring behavior, and API surface.

## Overview

This app helps users reflect on admissions decision strategy and modeled race disparity risk.

Current workflow:
1. Initializer page
2. Pairwise clarification submission
3. Trade-off graph with live analysis, explanation, and reflection capture

## Current System (Important)

- Active backend module: `race_disparity_pipeline`
- Do not run: `wisdom_stories_demo.app` (legacy path)
- Score scale: race disparity score is displayed on a `0-10` scale (not percent)

## Tech Stack

- Frontend: Vanilla HTML/CSS/JavaScript
- Backend: Flask + Flask-CORS
- Storage:
  - SQLite session store: `race_disparity_pipeline/study_sessions.db`
  - Reflection CSV export source: `race_disparity_pipeline/reflection_submissions.csv`

## Setup

1. Create virtual environment (if needed):

```powershell
python -m venv .venv
```

2. Activate virtual environment:

```powershell
& .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\race_disparity_pipeline\requirements.txt
```

## Run

Start the server with:

```powershell
.\.venv\Scripts\python.exe .\run_hci_server.py
```

Open:

- `http://127.0.0.1:5050/`
- or directly `http://127.0.0.1:5050/hybrid_initializer.html`

## End-to-End Flow

1. Open initializer.
2. Complete admissions decisions for at least 12 applicants (admit/reject + notes + overall rationale).
3. Submit pairwise clarifications.
4. Open the trade-off graph.
5. On graph update, UI now renders in this order:
   - Race disparity score and meter
   - Top disparity pairs table
   - Streaming AI explanation
6. Optionally submit reflection survey and export data.

## Scoring Behavior

- Race disparity score is model-derived and shown on a `0-10` scale.
- Higher value means larger modeled race-pair admit-probability gaps under matched synthetic comparisons.
- Common interpretation bands:
  - `0.0 - 1.2`: Low
  - `1.2 - 2.8`: Moderate
  - `2.8 - 4.5`: Elevated
  - `4.5 - 10.0`: High

## Core Files

- `hybrid_initializer.html`
- `tradeoffGraph.html`
- `run_hci_server.py`
- `race_disparity_pipeline/app.py`
- `race_disparity_pipeline/initialization_service.py`
- `race_disparity_pipeline/fairness_analysis.py`
- `race_disparity_pipeline/storage.py`

## API Endpoints

- `GET /api/health`
- `GET /api/init/default-applicants`
- `POST /api/init/start`
- `POST /api/init/answers`
- `POST /api/analysis/race-sensitivity`
- `POST /api/analysis/coach-insight`
- `POST /api/analysis/dialogue`
- `POST /api/analysis/apply-suggestion`
- `POST /api/analysis/ask-question`
- `POST /api/value-cards/recommend`
- `POST /api/events/confidence`
- `POST /api/events/reflection`
- `GET /api/export/sessions.csv`
- `GET /api/export/reflections.csv`

## Troubleshooting

1. If UI looks stale, hard refresh browser (`Ctrl+Shift+R`).
2. Ensure only one backend process is listening on `:5050`.
3. Verify health endpoint:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5050/api/health
```

4. Check active process on port 5050:

```powershell
Get-NetTCPConnection -LocalPort 5050 -State Listen |
  Select-Object -First 1 -ExpandProperty OwningProcess |
  ForEach-Object { Get-CimInstance Win32_Process -Filter "ProcessId=$($_)" | Select-Object ProcessId, Name, CommandLine }
```

Expected command line should include `run_hci_server.py` and `race_disparity_pipeline.app`.
