from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

try:
    from feature_extractor import extract_features
except ImportError:
    from .feature_extractor import extract_features


PAIRWISE_QUESTION_BANK = [
    {
        "id": "q_merit_context",
        "prompt": "When evidence is mixed, should committees prioritize merit evidence or family financial context?",
        "leftLabel": "Merit-based selection",
        "rightLabel": "Family financial context",
    },
    {
        "id": "q_consistency_flex",
        "prompt": "When interpretation is uncertain, should school resource context or community responsibility context carry more weight?",
        "leftLabel": "School resource context",
        "rightLabel": "Community responsibility context",
    },
    {
        "id": "q_efficiency_equity",
        "prompt": "Should your process lean more on merit-based selection or school resource context?",
        "leftLabel": "Merit-based selection",
        "rightLabel": "School resource context",
    },
    {
        "id": "q_research_grades",
        "prompt": "If two applicants are close, should family financial context or community responsibility context break the tie?",
        "leftLabel": "Family financial context",
        "rightLabel": "Community responsibility context",
    },
]

ANSWER_TO_SCORE = {
    "strong_left": -2,
    "left": -1,
    "balanced": 0,
    "right": 1,
    "strong_right": 2,
}

FEATURE_ORDER = [
    "gpa",
    "test_scores",
    "research",
    "leadership",
    "first_gen",
    "adversity",
    "community_impact",
    "gender_equity",
    "efficiency",
]


def _profile_feature_vector(profile: Dict[str, Any]) -> Dict[str, float]:
    gpa = float(profile.get("gpa", 3.0)) / 4.0
    test = float(profile.get("testScore", 1200.0)) / 1600.0
    research = float(profile.get("research", 0.0)) / 10.0
    leadership = float(profile.get("leadership", 0.0)) / 10.0
    adversity = float(profile.get("adversity", 0.0)) / 10.0
    community = float(profile.get("communityImpact", 0.0)) / 10.0
    first_gen = 1.0 if bool(profile.get("firstGen", False)) else 0.0
    review_complexity = float(profile.get("reviewComplexity", 5.0)) / 10.0

    return {
        "gpa": gpa,
        "test_scores": test,
        "research": research,
        "leadership": leadership,
        "first_gen": first_gen,
        "adversity": adversity,
        "community_impact": community,
        "gender_equity": 0.5 * community,
        "efficiency": review_complexity,
    }


def _build_regression_data(candidates: List[Dict[str, Any]], overall_rationale: str):
    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    notes_feature_scores: Dict[str, float] = {feature: 0.0 for feature in FEATURE_ORDER}

    for candidate in candidates:
        profile = candidate.get("profile") or {}
        notes = str(candidate.get("notes", ""))
        decision = str(candidate.get("decision", "")).strip().lower()
        signals = extract_features(notes)
        profile_features = _profile_feature_vector(profile)

        row = [1.0]
        for feature in FEATURE_ORDER:
            mention = 1.0 if signals[feature].present else 0.0
            signal_score = float(signals[feature].score)
            profile_score = profile_features.get(feature, 0.0)
            composite = 0.45 * mention + 0.25 * signal_score + 0.30 * profile_score
            row.append(composite)
            notes_feature_scores[feature] += signal_score

        X_rows.append(row)
        y_rows.append(1.0 if decision == "admit" else 0.0)

    rationale_signals = extract_features(overall_rationale)
    for feature in FEATURE_ORDER:
        notes_feature_scores[feature] += 0.6 * float(rationale_signals[feature].score)

    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float), notes_feature_scores


def _ridge_regression_weights(X: np.ndarray, y: np.ndarray, lam: float = 0.7) -> np.ndarray:
    n_features = X.shape[1]
    reg = lam * np.eye(n_features)
    reg[0, 0] = 0.0
    xtx = X.T @ X
    xty = X.T @ y
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + reg) @ xty


def _regression_to_feature_scores(coeffs: np.ndarray, notes_feature_scores: Dict[str, float]) -> Dict[str, float]:
    feature_scores: Dict[str, float] = {}
    for idx, feature in enumerate(FEATURE_ORDER, start=1):
        # Blend learned coefficient with mention strength so sparse notes remain interpretable.
        feature_scores[feature] = float(1.35 * coeffs[idx] + 0.18 * notes_feature_scores.get(feature, 0.0))
    return feature_scores


def _to_dimension_weights(feature_scores: Dict[str, float]) -> Dict[str, float]:
    # Semantic alignment to graph labels while preserving existing keys for compatibility:
    # - equity key -> Merit-Based Selection (left axis)
    # - individual key -> Family Financial Context (right axis)
    # - readiness key -> School Resource Context (top axis)
    # - opportunity key -> Community Responsibility Context (bottom axis)
    readiness = (
        1.0
        + 0.95 * feature_scores.get("test_scores", 0.0)
        + 0.85 * feature_scores.get("adversity", 0.0)
        + 0.55 * feature_scores.get("first_gen", 0.0)
    )
    opportunity = (
        1.0
        + 1.25 * feature_scores.get("community_impact", 0.0)
        + 0.55 * feature_scores.get("leadership", 0.0)
        + 0.45 * feature_scores.get("adversity", 0.0)
    )
    individual = (
        1.0
        + 1.20 * feature_scores.get("first_gen", 0.0)
        + 1.10 * feature_scores.get("adversity", 0.0)
        + 0.25 * feature_scores.get("community_impact", 0.0)
    )
    equity = (
        1.0
        + 1.15 * feature_scores.get("gpa", 0.0)
        + 0.95 * feature_scores.get("test_scores", 0.0)
        + 0.85 * feature_scores.get("research", 0.0)
        + 0.70 * feature_scores.get("leadership", 0.0)
    )

    clipped = {
        "readiness": max(0.05, readiness),
        "opportunity": max(0.05, opportunity),
        "individual": max(0.05, individual),
        "equity": max(0.05, equity),
    }
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def _ratio_pair(left: float, right: float) -> float:
    denom = left + right
    if denom <= 0:
        return 0.5
    return max(0.0, min(1.0, right / denom))


def _build_seed(weights: Dict[str, float]) -> Dict[str, float]:
    return {
        "xRatio": _ratio_pair(weights["equity"], weights["individual"]),
        "yRatio": _ratio_pair(weights["readiness"], weights["opportunity"]),
    }


def _candidate_vector_projection(candidates: List[Dict[str, Any]], feature_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    merit_features = ["gpa", "test_scores", "research", "leadership"]
    context_features = ["first_gen", "adversity", "community_impact"]

    vectors: List[Dict[str, Any]] = []
    for candidate in candidates:
        profile = candidate.get("profile") or {}
        profile_features = _profile_feature_vector(profile)

        merit = sum(profile_features.get(f, 0.0) * max(0.01, feature_scores.get(f, 0.01)) for f in merit_features)
        context = sum(profile_features.get(f, 0.0) * max(0.01, feature_scores.get(f, 0.01)) for f in context_features)
        x_axis = 0.5 + 0.3 * (feature_scores.get("gender_equity", 0.0) - feature_scores.get("efficiency", 0.0))
        y_axis = 0.5 + (context - merit)

        vectors.append(
            {
                "candidateLabel": candidate.get("candidateLabel", "Unknown"),
                "x": round(max(0.0, min(1.0, x_axis)), 3),
                "y": round(max(0.0, min(1.0, y_axis)), 3),
                "decision": candidate.get("decision", ""),
            }
        )

    return vectors


def generate_clarification_questions(weights: Dict[str, float], limit: int = 4) -> List[Dict[str, str]]:
    ranked = sorted(
        PAIRWISE_QUESTION_BANK,
        key=lambda item: 0 if item["id"] == "q_merit_context" else 1,
    )
    # Always include merit/context and efficiency/equity; fill with others.
    selected: List[Dict[str, str]] = []
    for question in ranked:
        if question["id"] in {"q_merit_context", "q_efficiency_equity"}:
            selected.append(question)
    for question in ranked:
        if question not in selected:
            selected.append(question)
        if len(selected) >= limit:
            break

    # If nearly balanced already, keep fewer questions to reduce user burden.
    balance_gap = abs(weights["readiness"] - weights["opportunity"]) + abs(weights["individual"] - weights["equity"])
    return selected[:3] if balance_gap < 0.2 else selected[:4]


def _adjust_weights_for_answers(weights: Dict[str, float], answers: Dict[str, str]) -> Dict[str, float]:
    adjusted = dict(weights)

    for question_id, option in answers.items():
        score = ANSWER_TO_SCORE.get(option, 0)
        if question_id == "q_merit_context":
            # Right option means stronger Family Financial Context over Merit-Based Selection.
            adjusted["equity"] += -0.035 * score
            adjusted["individual"] += 0.035 * score
        elif question_id == "q_consistency_flex":
            # Right option means stronger Community Responsibility Context over School Resource Context.
            adjusted["readiness"] += -0.032 * score
            adjusted["opportunity"] += 0.032 * score
        elif question_id == "q_efficiency_equity":
            # Right option means stronger School Resource Context over Merit-Based Selection.
            adjusted["equity"] += -0.030 * score
            adjusted["readiness"] += 0.030 * score
        elif question_id == "q_research_grades":
            # Right option means stronger Community Responsibility Context over Family Financial Context.
            adjusted["individual"] += -0.025 * score
            adjusted["opportunity"] += 0.025 * score

    clipped = {k: max(0.05, v) for k, v in adjusted.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def build_initial_snapshot(candidates: List[Dict[str, Any]], overall_rationale: str) -> Dict[str, object]:
    X, y, notes_feature_scores = _build_regression_data(candidates, overall_rationale)
    coeffs = _ridge_regression_weights(X, y)
    feature_scores = _regression_to_feature_scores(coeffs, notes_feature_scores)
    weights = _to_dimension_weights(feature_scores)
    questions = generate_clarification_questions(weights)
    seed = _build_seed(weights)
    candidate_vectors = _candidate_vector_projection(candidates, feature_scores)

    regression_details = {
        "intercept": round(float(coeffs[0]), 4),
        "coefficients": {
            feature: round(float(coeffs[idx]), 4)
            for idx, feature in enumerate(FEATURE_ORDER, start=1)
        },
        "method": "ridge_linear_probability",
        "lambda": 0.7,
    }

    return {
        "featureScores": {k: round(v, 4) for k, v in feature_scores.items()},
        "dimensionWeights": weights,
        "legacyDimensionWeights": {
            "merit": weights["readiness"],
            "context": weights["opportunity"],
            "gender": weights["individual"],
            "efficiency": weights["equity"],
        },
        "seed": seed,
        "clarificationQuestions": questions,
        "candidateVectors": candidate_vectors,
        "regressionDetails": regression_details,
    }


def finalize_weights(snapshot_weights: Dict[str, float], answers: Dict[str, str]) -> Dict[str, object]:
    final_weights = _adjust_weights_for_answers(snapshot_weights, answers)
    seed = _build_seed(final_weights)

    explanation = (
        "Your initial vector combines twelve admissions decisions, your written rationale, "
        "and targeted pairwise clarifications. Merit-based, family-financial, school-resource, and community-responsibility priorities were "
        "normalized into a single starting point for exploration."
    )

    return {
        "dimensionWeights": final_weights,
        "legacyDimensionWeights": {
            "merit": final_weights["readiness"],
            "context": final_weights["opportunity"],
            "gender": final_weights["individual"],
            "efficiency": final_weights["equity"],
        },
        "seed": seed,
        "explanation": explanation,
    }
