from __future__ import annotations

import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


DB_PATH = Path(__file__).resolve().parent / "study_sessions.db"
REFLECTION_TABLE_PATH = Path(__file__).resolve().parent / "reflection_submissions.csv"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                snapshot_json TEXT,
                final_json TEXT
            );

            CREATE TABLE IF NOT EXISTS pairwise_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                answer_value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def create_session(payload: Dict[str, Any], snapshot: Dict[str, Any]) -> str:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(session_id, created_at, payload_json, snapshot_json) VALUES (?, ?, ?, ?)",
            (session_id, created_at, json.dumps(payload), json.dumps(snapshot)),
        )
    return session_id


def save_answers(session_id: str, answers: Dict[str, str]) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    rows = [(session_id, qid, answer, created_at) for qid, answer in answers.items()]
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO pairwise_answers(session_id, question_id, answer_value, created_at) VALUES (?, ?, ?, ?)",
            rows,
        )


def save_final(session_id: str, final_payload: Dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET final_json = ? WHERE session_id = ?",
            (json.dumps(final_payload), session_id),
        )


def log_event(event_type: str, event_payload: Dict[str, Any], session_id: str | None = None) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events(session_id, event_type, event_json, created_at) VALUES (?, ?, ?, ?)",
            (session_id, event_type, json.dumps(event_payload), created_at),
        )


def get_snapshot(session_id: str) -> Dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT snapshot_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row or not row["snapshot_json"]:
        return None
    return json.loads(row["snapshot_json"])


def get_session_payload(session_id: str) -> Dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT payload_json FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not row or not row["payload_json"]:
        return None
    return json.loads(row["payload_json"])


def export_sessions() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.created_at, s.payload_json, s.snapshot_json, s.final_json
            FROM sessions s
            ORDER BY s.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def export_reflections() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.session_id, e.event_json, e.created_at
            FROM events e
            WHERE e.event_type = 'reflection'
            ORDER BY e.created_at DESC
            """
        ).fetchall()

    result: List[Dict[str, Any]] = []
    for row in rows:
        raw_json = row["event_json"] or "{}"
        try:
            payload = json.loads(raw_json)
        except Exception:
            payload = {}

        responses = payload.get("responses") if isinstance(payload, dict) else {}
        if not isinstance(responses, dict):
            responses = {}

        result.append(
            {
                "event_id": row["id"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "surprise_level": responses.get("surpriseLevel", ""),
                "seen_similar_tool_before": responses.get("seenSimilarToolBefore", ""),
                "usefulness_rating": responses.get("usefulnessRating", ""),
                "recommend_rating": responses.get("recommendRating", ""),
                "change_intent_rating": responses.get("changeIntentRating", ""),
                "reflection_text": responses.get("reflectionText", ""),
                "submitted_at": responses.get("submittedAt", ""),
            }
        )

    return result


def append_reflection_submission(session_id: str | None, responses: Dict[str, Any]) -> Path:
    file_exists = REFLECTION_TABLE_PATH.exists()
    created_at = datetime.now(timezone.utc).isoformat()

    REFLECTION_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFLECTION_TABLE_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "created_at",
                    "session_id",
                    "surprise_level",
                    "seen_similar_tool_before",
                    "usefulness_rating",
                    "recommend_rating",
                    "change_intent_rating",
                    "reflection_text",
                    "submitted_at",
                ]
            )

        writer.writerow(
            [
                created_at,
                session_id or "",
                responses.get("surpriseLevel", ""),
                responses.get("seenSimilarToolBefore", ""),
                responses.get("usefulnessRating", ""),
                responses.get("recommendRating", ""),
                responses.get("changeIntentRating", ""),
                responses.get("reflectionText", ""),
                responses.get("submittedAt", ""),
            ]
        )

    return REFLECTION_TABLE_PATH
