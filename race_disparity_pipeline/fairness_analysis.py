from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .feature_extractor import extract_features
except Exception:
    try:
        from feature_extractor import extract_features
    except Exception:
        extract_features = None


VALUE_KEYS = ["merit", "family", "school", "community"]
DEFAULT_RACES = ["Asian", "Black", "Latinx", "White"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_value_weights(seed: Dict[str, Any]) -> Dict[str, float]:
    raw = seed.get("valueWeights") if isinstance(seed, dict) else None
    if isinstance(raw, dict):
        vals = {
            "merit": max(0.0, float(raw.get("merit", 0.0))),
            "family": max(0.0, float(raw.get("family", 0.0))),
            "school": max(0.0, float(raw.get("school", 0.0))),
            "community": max(0.0, float(raw.get("community", 0.0))),
        }
        total = sum(vals.values())
        if total > 0:
            return {k: vals[k] / total for k in VALUE_KEYS}

    x_ratio = float(seed.get("xRatio", 0.5)) if isinstance(seed, dict) else 0.5
    y_ratio = float(seed.get("yRatio", 0.5)) if isinstance(seed, dict) else 0.5
    return {
        "merit": max(0.0, 1.0 - x_ratio),
        "family": max(0.0, x_ratio),
        "school": max(0.0, 1.0 - y_ratio),
        "community": max(0.0, y_ratio),
    }


def _seed_from_value_weights(weights: Dict[str, float]) -> Dict[str, float]:
    horizontal = float(weights.get("merit", 0.0)) + float(weights.get("family", 0.0))
    vertical = float(weights.get("school", 0.0)) + float(weights.get("community", 0.0))
    x_ratio = 0.5 if horizontal <= 0 else float(weights.get("family", 0.0)) / horizontal
    y_ratio = 0.5 if vertical <= 0 else float(weights.get("community", 0.0)) / vertical
    return {
        "xRatio": _clamp(x_ratio, 0.0, 1.0),
        "yRatio": _clamp(y_ratio, 0.0, 1.0),
    }


def _profile_components(profile: Dict[str, Any]) -> Dict[str, float]:
    return {
        "gpa": _clamp(float(profile.get("gpa", 3.0)) / 4.0, 0.0, 1.0),
        "test": _clamp(float(profile.get("testScore", 1200.0)) / 1600.0, 0.0, 1.0),
        "research": _clamp(float(profile.get("research", 0.0)) / 10.0, 0.0, 1.0),
        "leadership": _clamp(float(profile.get("leadership", 0.0)) / 10.0, 0.0, 1.0),
        "adversity": _clamp(float(profile.get("adversity", 0.0)) / 10.0, 0.0, 1.0),
        "community": _clamp(float(profile.get("communityImpact", 0.0)) / 10.0, 0.0, 1.0),
        "first_gen": 1.0 if bool(profile.get("firstGen", False)) else 0.0,
    }


def _nonrace_score(profile: Dict[str, Any], value_weights: Dict[str, float]) -> float:
    c = _profile_components(profile)
    merit_side = 0.45 * c["gpa"] + 0.30 * c["test"] + 0.15 * c["research"] + 0.10 * c["leadership"]
    family_side = 0.55 * c["adversity"] + 0.45 * c["first_gen"]
    school_side = 0.60 * c["adversity"] + 0.25 * c["community"] + 0.15 * c["first_gen"]
    community_side = 0.65 * c["community"] + 0.25 * c["leadership"] + 0.10 * c["adversity"]

    return (
        float(value_weights.get("merit", 0.25)) * merit_side
        + float(value_weights.get("family", 0.25)) * family_side
        + float(value_weights.get("school", 0.25)) * school_side
        + float(value_weights.get("community", 0.25)) * community_side
    )


def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _learn_personalized_model(candidates: List[Dict[str, Any]], value_weights: Dict[str, float]) -> Dict[str, Any]:
    observed_races = sorted({str((c.get("profile") or {}).get("raceGroup", "Unknown")) for c in candidates})
    race_groups = [r for r in observed_races if r and r != "Unknown"]
    if len(race_groups) < 2:
        race_groups = list(DEFAULT_RACES)

    base_race = race_groups[0]
    race_to_idx = {race: idx for idx, race in enumerate(race_groups)}

    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    for candidate in candidates:
        profile = candidate.get("profile") or {}
        race = str(profile.get("raceGroup", base_race))
        score = _nonrace_score(profile, value_weights)

        row = [1.0, score]
        for race_name in race_groups[1:]:
            row.append(1.0 if race == race_name else 0.0)

        decision = str(candidate.get("decision", "")).strip().lower()
        y = 1.0 if decision == "admit" else 0.0

        X_rows.append(row)
        y_rows.append(y)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)

    lam = 1.5
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    try:
        coef = np.linalg.solve(X.T @ X + reg, X.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(X.T @ X + reg) @ (X.T @ y)

    intercept = float(coef[0])
    base_scale = float(coef[1])

    race_terms: Dict[str, float] = {base_race: 0.0}
    for idx, race_name in enumerate(race_groups[1:], start=2):
        race_terms[race_name] = _clamp(float(coef[idx]), -0.35, 0.35)

    return {
        "valueWeights": value_weights,
        "raceGroups": race_groups,
        "raceToIdx": race_to_idx,
        "intercept": intercept,
        "baseScale": base_scale,
        "raceTerms": race_terms,
    }


def _predict_admit_probability(profile: Dict[str, Any], race: str, model: Dict[str, Any]) -> float:
    base = _nonrace_score(profile, model["valueWeights"])
    race_term = float(model["raceTerms"].get(race, 0.0))
    logit = float(model["intercept"]) + float(model["baseScale"]) * base + race_term
    return _clamp(_sigmoid(logit), 0.0, 1.0)


def _generate_benchmark_sets(model: Dict[str, Any], num_sets: int = 100) -> List[List[Dict[str, Any]]]:
    rng = np.random.default_rng(42)
    races = list(model.get("raceGroups") or DEFAULT_RACES)
    if len(races) < 4:
        for race in DEFAULT_RACES:
            if race not in races:
                races.append(race)
            if len(races) == 4:
                break
    races = races[:4]

    sets: List[List[Dict[str, Any]]] = []
    for set_idx in range(num_sets):
        base_profile = {
            "gpa": float(rng.uniform(2.6, 4.0)),
            "testScore": float(rng.uniform(980, 1600)),
            "research": float(rng.uniform(0.0, 10.0)),
            "leadership": float(rng.uniform(0.0, 10.0)),
            "adversity": float(rng.uniform(0.0, 10.0)),
            "communityImpact": float(rng.uniform(0.0, 10.0)),
            "firstGen": bool(rng.random() < 0.35),
            "reviewComplexity": float(rng.uniform(2.0, 10.0)),
        }

        group: List[Dict[str, Any]] = []
        for race in races:
            # Small jitter keeps candidates highly comparable but not perfectly identical.
            profile = {
                "gpa": _clamp(base_profile["gpa"] + float(rng.normal(0.0, 0.05)), 2.0, 4.0),
                "testScore": _clamp(base_profile["testScore"] + float(rng.normal(0.0, 35.0)), 800.0, 1600.0),
                "research": _clamp(base_profile["research"] + float(rng.normal(0.0, 0.35)), 0.0, 10.0),
                "leadership": _clamp(base_profile["leadership"] + float(rng.normal(0.0, 0.35)), 0.0, 10.0),
                "adversity": _clamp(base_profile["adversity"] + float(rng.normal(0.0, 0.35)), 0.0, 10.0),
                "communityImpact": _clamp(base_profile["communityImpact"] + float(rng.normal(0.0, 0.35)), 0.0, 10.0),
                "firstGen": base_profile["firstGen"],
                "reviewComplexity": _clamp(base_profile["reviewComplexity"] + float(rng.normal(0.0, 0.25)), 1.0, 10.0),
                "raceGroup": race,
            }
            group.append({
                "setId": set_idx,
                "raceGroup": race,
                "profile": profile,
            })
        sets.append(group)

    return sets


def _pair_disparity_from_sets(matched_sets: List[List[Dict[str, Any]]], model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute disparity using one max-gap comparison per applicant in each matched set.

    For each set, we generate 4 race variants of comparable credentials and predict admit probability for each.
    Then, for each applicant, we compare its predicted probability to the other 3 applicants in that matched set
    and keep only the largest absolute gap for that applicant.

    This yields up to 4 recorded gaps per set and matches the intended "max gap per applicant" aggregation rule.
    """
    pair_records: Dict[Tuple[str, str], List[Dict[str, float]]] = {}

    for group in matched_sets:
        scored = []
        for row in group:
            race = str(row["raceGroup"])
            profile = row["profile"]
            prob = _predict_admit_probability(profile, race, model)
            scored.append({"race": race, "prob": prob})

        for i in range(len(scored)):
            race_i = scored[i]["race"]
            prob_i = float(scored[i]["prob"])

            best_gap = -1.0
            best_signed = 0.0
            best_pair_key: Tuple[str, str] | None = None

            for j in range(len(scored)):
                if i == j:
                    continue

                race_j = scored[j]["race"]
                prob_j = float(scored[j]["prob"])
                pair_key = _canonical_pair(race_i, race_j)
                gap = abs(prob_i - prob_j)
                signed = prob_i - prob_j if pair_key[0] == race_i else prob_j - prob_i

                if gap > best_gap:
                    best_gap = gap
                    best_signed = signed
                    best_pair_key = pair_key

            if best_pair_key is not None:
                pair_records.setdefault(best_pair_key, []).append({
                    "gap": float(best_gap),
                    "signed": float(best_signed),
                })

    pair_stats = []
    for pair_key, rows in pair_records.items():
        gaps = [float(r["gap"]) for r in rows]
        signed = [float(r["signed"]) for r in rows]
        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        mean_signed = float(np.mean(signed)) if signed else 0.0
        if mean_signed > 0:
            favored = pair_key[0]
        elif mean_signed < 0:
            favored = pair_key[1]
        else:
            favored = "tie"

        pair_stats.append(
            {
                "pairKey": f"{pair_key[0]} vs {pair_key[1]}",
                "races": [pair_key[0], pair_key[1]],
                "meanGap": round(mean_gap, 4),
                "meanSignedGap": round(mean_signed, 4),
                "favoredRace": favored,
                "count": len(rows),
            }
        )

    pair_stats.sort(key=lambda item: item["meanGap"], reverse=True)
    overall = 100.0 * float(pair_stats[0]["meanGap"]) if pair_stats else 0.0

    return {
        "overallScore": round(overall, 1),
        "pairStats": pair_stats,
        "topPairs": pair_stats[:3],
        "rawRecords": pair_records,
    }


def _value_action_consistency(overall_rationale: str, value_weights: Dict[str, float]) -> Dict[str, Any]:
    text = str(overall_rationale or "").strip()
    default_weights = {"merit": 0.25, "family": 0.25, "school": 0.25, "community": 0.25}
    if not text or extract_features is None:
        return {
            "rationaleWeights": default_weights,
            "note": "No rationale text features available for consistency comparison.",
        }

    signals = extract_features(text)
    merit_signal = (
        float(signals["gpa"].score)
        + float(signals["test_scores"].score)
        + float(signals["research"].score)
        + float(signals["leadership"].score)
    ) / 4.0
    family_signal = (float(signals["first_gen"].score) + float(signals["adversity"].score)) / 2.0
    school_signal = (
        float(signals["adversity"].score)
        + float(signals["efficiency"].score)
        + float(signals["first_gen"].score)
    ) / 3.0
    community_signal = (
        float(signals["community_impact"].score)
        + float(signals["leadership"].score)
        + float(signals["adversity"].score)
    ) / 3.0

    rationale_raw = {
        "merit": max(0.01, merit_signal),
        "family": max(0.01, family_signal),
        "school": max(0.01, school_signal),
        "community": max(0.01, community_signal),
    }
    raw_total = sum(rationale_raw.values())
    rationale_weights = {k: rationale_raw[k] / raw_total for k in VALUE_KEYS}

    note = "Rationale weights extracted successfully for reference."

    return {
        "rationaleWeights": {k: round(rationale_weights[k], 4) for k in VALUE_KEYS},
        "note": note,
    }


def compute_counterfactual_race_sensitivity(candidates: List[Dict[str, Any]], seed: Dict[str, float], overall_rationale: str = "") -> Dict[str, Any]:
    if not candidates:
        return {
            "raceSensitivity": 0.0,
            "disparityScore": 0.0,
            "topDisparityPairs": [],
            "pairDisparities": [],
            "counterfactualChanges": [],
            "counterfactualPromptTemplate": "No candidates available for analysis.",
            "valueWeights": {k: 0.25 for k in VALUE_KEYS},
            "seed": {"xRatio": 0.5, "yRatio": 0.5},
        }

    value_weights = _normalize_value_weights(seed)
    import sys
    print(f"[compute_counterfactual] Seed input: {seed}", file=sys.stderr, flush=True)
    print(f"[compute_counterfactual] Normalized valueWeights: {value_weights}", file=sys.stderr, flush=True)
    seed_view = _seed_from_value_weights(value_weights)
    model = _learn_personalized_model(candidates, value_weights)

    benchmark_sets = _generate_benchmark_sets(model, num_sets=100)
    disparity = _pair_disparity_from_sets(benchmark_sets, model)

    top_pairs = disparity["topPairs"]
    counterfactual_changes = [
        {
            "candidateLabel": item["pairKey"],
            "raceGroup": item["favoredRace"],
            "maxDecisionShift": item["meanGap"],
            "perRaceShift": {
                item["races"][0]: round(max(0.0, item["meanSignedGap"]), 4),
                item["races"][1]: round(max(0.0, -item["meanSignedGap"]), 4),
            },
        }
        for item in top_pairs
    ]

    prompt_template = (
        "Race Disparity Score is computed on a 400-profile synthetic benchmark (100 matched sets x 4 races). "
        "Within each matched set, we compute each applicant's predicted admit probability, compare that applicant to the other 3 in the set, "
        "keep the single largest probability gap for that applicant, then aggregate those per-applicant winners by canonical race pair and report 100 x the highest mean pair gap."
    )

    raw_score = float(disparity["overallScore"])
    display_score = _rescale_sensitivity(raw_score)

    return {
        "raceSensitivity": round(display_score, 2),
        "rawRaceSensitivity": round(raw_score, 1),
        "disparityScore": round(display_score, 2),
        "dpGap": round(display_score / 100.0, 4),
        "adverseImpactRatio": round(max(0.0, 1.0 - display_score / 100.0), 4),
        "topDisparityPairs": top_pairs,
        "pairDisparities": disparity["pairStats"],
        "benchmarkSets": len(benchmark_sets),
        "benchmarkApplicants": len(benchmark_sets) * 4,
        "valueWeights": {k: round(value_weights[k], 4) for k in VALUE_KEYS},
        "seed": seed_view,
        "counterfactualChanges": counterfactual_changes,
        "counterfactualPromptTemplate": prompt_template,
        "modelDiagnostics": {
            "intercept": round(model["intercept"], 4),
            "baseScale": round(model["baseScale"], 4),
            "raceTerms": {k: round(v, 4) for k, v in model["raceTerms"].items()},
        },
        "leaveOneOut": {
            "values": [],
            "range": 0.0,
            "std": 0.0,
            "mean": 0.0,
            "confidenceLabel": "n/a",
            "confidenceNote": "Replaced by matched-set benchmark aggregation.",
        },
    }


def suggest_lower_sensitivity_target(candidates: List[Dict[str, Any]], seed: Dict[str, float]) -> Dict[str, Any]:
    # Legacy compatibility endpoint. The UI no longer uses suggestion cards.
    current_weights = _normalize_value_weights(seed)
    current_seed = _seed_from_value_weights(current_weights)
    baseline = compute_counterfactual_race_sensitivity(candidates, {"valueWeights": current_weights, **current_seed})
    return {
        "currentSeed": current_seed,
        "currentValueWeights": {k: round(current_weights[k], 4) for k in VALUE_KEYS},
        "currentRaceSensitivity": float(baseline.get("raceSensitivity", 0.0)),
        "suggestedSeed": current_seed,
        "suggestedValueWeights": {k: round(current_weights[k], 4) for k in VALUE_KEYS},
        "suggestedRaceSensitivity": float(baseline.get("raceSensitivity", 0.0)),
        "estimatedReduction": 0.0,
    }


def generate_tiered_suggestions(seed: Dict[str, float], sensitivity: float, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Legacy compatibility endpoint. Intentionally empty in the new flow.
    return {
        "suggestions": [],
        "currentSensitivity": round(float(sensitivity), 1),
    }


def _clean_ai_text(text: str) -> str:
    return str(text or "").replace("—", ", ").replace("\u2014", ", ")


def _rescale_sensitivity(raw_score: float) -> float:
    """Return the raw benchmark score on 0-10 display scale."""
    score = max(0.0, min(100.0, float(raw_score)))
    return score


def generate_initialization_explanation(dimension_weights: Dict[str, float], race_sensitivity: float) -> str:
    school_resource = float(dimension_weights.get("readiness", 0.25)) * 100
    community_resp = float(dimension_weights.get("opportunity", 0.25)) * 100
    family_financial = float(dimension_weights.get("individual", 0.25)) * 100
    merit = float(dimension_weights.get("equity", 0.25)) * 100

    return (
        f'<div style="line-height:1.45;font-size:0.93rem;">'
        f'<p><strong>What we computed:</strong> We used your 12 admit/reject decisions, notes, and pairwise clarifications to estimate your decision policy.</p>'
        f'<p><strong>Your policy mix:</strong> Merit <strong>{merit:.0f}%</strong>, Family Context <strong>{family_financial:.0f}%</strong>, '
        f'School Context <strong>{school_resource:.0f}%</strong>, Community Context <strong>{community_resp:.0f}%</strong>.</p>'
        f'<p><strong>Race Disparity Score:</strong> <strong>{race_sensitivity:.1f}</strong>. '
        f'This is model-based risk from a matched synthetic benchmark.</p>'
        f'</div>'
    )


def generate_plain_explanation(seed: Dict[str, float], sensitivity: float, weights: Dict[str, float] | None = None) -> str:
    if weights:
        merit_pct = float(weights.get("equity", 0.25)) * 100
        family_pct = float(weights.get("individual", 0.25)) * 100
        school_pct = float(weights.get("readiness", 0.25)) * 100
        community_pct = float(weights.get("opportunity", 0.25)) * 100
        return _clean_ai_text(
            f"Your current policy emphasizes Merit {merit_pct:.0f}%, Family {family_pct:.0f}%, "
            f"School {school_pct:.0f}%, and Community {community_pct:.0f}%. "
            f"The current model-based race disparity score is {float(sensitivity):.1f}."
        )

    x = float(seed.get("xRatio", 0.5))
    y = float(seed.get("yRatio", 0.5))
    return _clean_ai_text(
        f"Your current stance balances Merit vs Family at x={x:.2f} and School vs Community at y={y:.2f}. "
        f"Current model-based race disparity score is {float(sensitivity):.1f}."
    )
