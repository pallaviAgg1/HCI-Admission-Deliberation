from __future__ import annotations

from typing import Any, Dict, List


VALUE_CARDS: List[Dict[str, Any]] = [
    {
        "id": "socio_opportunity",
        "title": "Socioeconomic Opportunity",
        "description": "Treat opportunity as part of merit, not a separate charity lens.",
        "attentionalPolicy": [
            "First-generation status: what barriers did they navigate without inherited playbooks?",
            "School resources: AP courses, lab access, and counselor support available to them.",
            "Work obligations: paid work load can suppress GPA while building resilience.",
            "Family responsibilities: caregiving and financial support can limit extracurricular options.",
            "Summer pathway: paid jobs versus unpaid internships often reflect opportunity gaps.",
        ],
        "suggestedShift": {"xDelta": 0.02, "yDelta": 0.09},
        "tags": ["context", "first_gen", "adversity"],
    },
    {
        "id": "evidence_quality",
        "title": "Evidence Quality",
        "description": "Favor evidence depth over single-score certainty.",
        "attentionalPolicy": [
            "Look for multi-source evidence: coursework, recommendations, and sustained projects.",
            "Distinguish one-time performance from repeated competence.",
            "Check if high scores were achieved with unusual preparation advantages.",
            "Note upward trend in performance when context improved.",
            "Ask whether your confidence depends on one metric only.",
        ],
        "suggestedShift": {"xDelta": 0.0, "yDelta": -0.05},
        "tags": ["merit", "test_scores", "gpa"],
    },
    {
        "id": "cohort_equity",
        "title": "Cohort Equity and Representation",
        "description": "Consider whether your cohort-building lens systematically filters out specific groups.",
        "attentionalPolicy": [
            "Ask which applicant groups disappear when your rubric is applied strictly.",
            "Check if efficiency shortcuts over-penalize complex but promising files.",
            "Look for talent signals that traditional metrics under-capture.",
            "Ensure risk language is applied consistently across demographic groups.",
            "Separate 'familiar profile' preference from evidence of future contribution.",
        ],
        "suggestedShift": {"xDelta": 0.08, "yDelta": 0.03},
        "tags": ["gender_equity", "representation", "fairness"],
    },
    {
        "id": "efficiency_guardrail",
        "title": "Efficiency Guardrail",
        "description": "Use speed as a guardrail, not a default decision-maker.",
        "attentionalPolicy": [
            "Flag files where short review time could miss context-heavy evidence.",
            "Define a threshold for when to escalate to deeper read.",
            "Prioritize consistency checks for borderline candidates.",
            "Track whether fast decisions correlate with lower diversity outcomes.",
            "Log rationale completeness before finalizing a decision.",
        ],
        "suggestedShift": {"xDelta": -0.08, "yDelta": 0.01},
        "tags": ["efficiency", "process", "consistency"],
    },
]


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, value))


def recommend_value_cards(seed: Dict[str, float], limit: int = 3) -> List[Dict[str, Any]]:
    x_ratio = float(seed.get("xRatio", 0.5))
    y_ratio = float(seed.get("yRatio", 0.5))

    scored = []
    for card in VALUE_CARDS:
        y_target = _clamp_ratio(y_ratio + card["suggestedShift"]["yDelta"])
        x_target = _clamp_ratio(x_ratio + card["suggestedShift"]["xDelta"])
        impact = abs(y_target - y_ratio) + abs(x_target - x_ratio)

        relevance = 0.2 + impact
        if card["id"] == "socio_opportunity" and y_ratio < 0.55:
            relevance += 0.3
        if card["id"] == "cohort_equity" and x_ratio < 0.55:
            relevance += 0.25
        if card["id"] == "efficiency_guardrail" and x_ratio < 0.4:
            relevance += 0.2

        scored.append((relevance, card, x_target, y_target))

    scored.sort(key=lambda item: item[0], reverse=True)

    cards = []
    for _, card, x_target, y_target in scored[:limit]:
        cards.append(
            {
                "id": card["id"],
                "title": card["title"],
                "description": card["description"],
                "attentionalPolicy": card["attentionalPolicy"],
                "suggestedShift": card["suggestedShift"],
                "suggestedTarget": {"xRatio": round(x_target, 3), "yRatio": round(y_target, 3)},
                "tags": card["tags"],
            }
        )

    return cards
