# Admissions Reflection Prototype

Single guide for setup, usage, and technical behavior.

## What This App Does

The app helps you reflect on admissions trade-offs and bias risk in two steps:

1. Initializer:
- Evaluate 6 applicants (admit/reject)
- Add notes + overall rationale
- Answer pairwise clarification questions
- Produce a seeded trade-off point

2. Trade-Off Graph:
- Drag your point on a 2D map
- See race sensitivity update in real time
- Try Conservative / Moderate / Ambitious suggestions

## Tech Stack

- Frontend: Vanilla HTML/CSS/JavaScript
- Backend: Flask + Flask-CORS
 Data: CSV + SQLite (`race_disparity_pipeline/study_sessions.db`)
 .\venv\Scripts\python.exe -m pip install -r .\requirements.txt
 .\venv\Scripts\python.exe -m race_disparity_pipeline.app
 It is a custom in-code counterfactual simulation implemented in `race_disparity_pipeline/fairness_analysis.py`.
 `race_disparity_pipeline/app.py`
 `race_disparity_pipeline/initialization_service.py`
 `race_disparity_pipeline/fairness_analysis.py`
 `race_disparity_pipeline/storage.py`
```powershell
& .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\wisdom_stories_demo\requirements.txt
```

3. Run server:

```powershell
.\.venv\Scripts\python.exe -m wisdom_stories_demo.app
```

4. Open:

`http://127.0.0.1:5050/`

## End-to-End Flow

1. Open the initializer. You will first see a short study primer that explains the six-applicant admissions task and the race-sensitivity check.
2. Complete 6 applicant decisions + notes + overall rationale.
3. Generate and answer pairwise questions.
4. Open seeded trade-off graph.
5. Explore point movement or choose suggestion cards.
6. (Optional) export sessions from `/api/export/sessions.csv`.

## Pairwise Questions (Current)

- Merit-Based Selection vs Family Financial Context
- Merit-Based Selection vs School Resource Context
- School Resource Context vs Community Responsibility Context
- Family Financial Context vs Community Responsibility Context

Depending on your current balance, the app may show 3 or 4 questions.

## CPF (Counterfactual Fairness) in Plain English

Technology used:
- No external CPF library/tool is used.
- It is a custom in-code counterfactual simulation implemented in `wisdom_stories_demo/fairness_analysis.py`.

How it works:
1. At your current point, it computes each applicant's score.
2. For that same applicant, it swaps only race group and recomputes score across configured race groups.
3. It measures score shift from those swaps and aggregates into one race-sensitivity value.

How suggestions are generated around your point:
- It checks a nearby interior grid of candidate points from `x=0.10..0.90` and `y=0.10..0.90` in `0.05` steps.
- That is `17 x 17 = 289` candidate points.
- For each candidate point, it reruns the same CPF sensitivity calculation.
- It selects the best target by minimizing:
  - CPF sensitivity
  - movement distance from your current point
  - edge penalty (avoid brittle edge/corner solutions)
- Then creates:
  - Conservative = 35% move to target
  - Moderate = 70% move to target
  - Ambitious = 100% move to target

## Core Files

- `hybrid_initializer.html`
- `tradeoffGraph.html`
- `wisdom_stories_demo/app.py`
- `wisdom_stories_demo/initialization_service.py`
- `wisdom_stories_demo/fairness_analysis.py`
- `wisdom_stories_demo/storage.py`

## API Endpoints

- `GET /api/health`
- `POST /api/init/start`
- `POST /api/init/answers`
- `POST /api/analysis/dialogue`
- `POST /api/analysis/race-sensitivity`
- `POST /api/events/confidence`
- `GET /api/export/sessions.csv`

## Quick Troubleshooting

- If you see stale question text, hard refresh the browser (`Ctrl+F5`).
- Ensure only one backend process is running on `:5050`.
- Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5050/api/health
```
