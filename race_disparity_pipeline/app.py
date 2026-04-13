from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, Response, send_from_directory, redirect
from flask_cors import CORS

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from .initialization_service import build_initial_snapshot, finalize_weights
    from .storage import (
        append_reflection_submission,
        create_session,
        export_reflections,
        export_sessions,
        get_session_payload,
        get_snapshot,
        init_db,
        log_event,
        save_answers,
        save_final,
    )
    from .fairness_analysis import (
        compute_counterfactual_race_sensitivity,
        suggest_lower_sensitivity_target,
        generate_plain_explanation,
        generate_tiered_suggestions,
        generate_initialization_explanation,
    )
    from .sample_applicants import get_default_applicants
    from .value_cards import recommend_value_cards
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from initialization_service import build_initial_snapshot, finalize_weights
    from storage import (
        append_reflection_submission,
        create_session,
        export_reflections,
        export_sessions,
        get_session_payload,
        get_snapshot,
        init_db,
        log_event,
        save_answers,
        save_final,
    )
    from fairness_analysis import (
        compute_counterfactual_race_sensitivity,
        suggest_lower_sensitivity_target,
        generate_plain_explanation,
        generate_tiered_suggestions,
        generate_initialization_explanation,
    )
    from sample_applicants import get_default_applicants
    from value_cards import recommend_value_cards


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT.parent

app = Flask(__name__)
CORS(app)
init_db()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAPIKEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _build_weight_snapshot(seed: Dict[str, float], weights: Dict[str, float] | None = None) -> Dict[str, float]:
    if weights:
        if all(key in weights for key in ("merit", "family", "school", "community")):
            return {
                "merit": float(weights.get("merit", 0.0)) * 100.0,
                "family": float(weights.get("family", 0.0)) * 100.0,
                "school": float(weights.get("school", 0.0)) * 100.0,
                "community": float(weights.get("community", 0.0)) * 100.0,
            }
        return {
            "merit": float(weights.get("equity", 0.0)) * 100.0,
            "family": float(weights.get("individual", 0.0)) * 100.0,
            "school": float(weights.get("readiness", 0.0)) * 100.0,
            "community": float(weights.get("opportunity", 0.0)) * 100.0,
        }
    if isinstance(seed.get("valueWeights"), dict):
        vw = seed.get("valueWeights") or {}
        return {
            "merit": float(vw.get("merit", 0.0)) * 100.0,
            "family": float(vw.get("family", 0.0)) * 100.0,
            "school": float(vw.get("school", 0.0)) * 100.0,
            "community": float(vw.get("community", 0.0)) * 100.0,
        }
    x_ratio = float(seed.get("xRatio", 0.5))
    y_ratio = float(seed.get("yRatio", 0.5))
    return {
        "merit": (1.0 - x_ratio) * 100.0,
        "family": x_ratio * 100.0,
        "school": (1.0 - y_ratio) * 100.0,
        "community": y_ratio * 100.0,
    }


def _llm_chat_response(system_prompt: str, user_prompt: str, temperature: float = 0.35, max_tokens: int = 260) -> str | None:
    if not OPENAI_API_KEY or OpenAI is None:
        print("[llm] OpenAI unavailable (missing key or SDK). Using fallback response.", flush=True)
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = (completion.choices[0].message.content or "").strip()
        print(f"[llm] OpenAI response received. model={OPENAI_CHAT_MODEL}, has_content={bool(content)}", flush=True)
        return content or None
    except Exception as exc:
        print(f"[llm] OpenAI request failed. Using fallback. error={exc}", flush=True)
        return None


def _llm_stance_explanation(seed: Dict[str, float], sensitivity: float, weights: Dict[str, float] | None = None) -> str | None:
    summary = _build_weight_snapshot(seed, weights)
    prompt = (
        "Explain the user's current admissions stance in 3-4 short sentences. "
        "Be beginner-friendly and concrete, avoid jargon, and avoid repeating the sensitivity value twice. "
        "Use these values:\n"
        f"- Merit-Based Selection: {summary['merit']:.1f}%\n"
        f"- Family Financial Context: {summary['family']:.1f}%\n"
        f"- School Resource Context: {summary['school']:.1f}%\n"
        f"- Community Responsibility Context: {summary['community']:.1f}%\n"
        f"- Current race sensitivity: {sensitivity:.1f}%\n"
        "Mention one practical trade-off implied by this stance."
    )
    return _llm_chat_response(
        "You are a concise, supportive admissions fairness coach.",
        prompt,
        temperature=0.3,
        max_tokens=220,
    )


def _llm_question_answer(question: str, seed: Dict[str, float], sensitivity: float, fairness: Dict[str, Any], weights: Dict[str, float] | None = None) -> str | None:
    summary = _build_weight_snapshot(seed, weights)
    top_shift = 0.0
    if fairness.get("counterfactualChanges"):
        top_shift = max(float(item.get("maxDecisionShift", 0.0)) for item in fairness.get("counterfactualChanges", []))

    prompt = (
        "Answer the user's question about their admissions stance and bias metric in plain English. "
        "Be specific, 3-6 sentences, and reference the numbers below when useful. "
        "Do not make absolute claims like 'no bias' or 'definitely biased'. "
        "Interpret score bands as: 0-1.2 low, 1.2-2.8 moderate, 2.8-4.5 elevated, 4.5-10 high modeled disparity risk.\n\n"
        f"User question: {question}\n"
        f"Current race disparity risk score: {sensitivity:.1f}\n"
        f"Merit-Based Selection: {summary['merit']:.1f}%\n"
        f"Family Financial Context: {summary['family']:.1f}%\n"
        f"School Resource Context: {summary['school']:.1f}%\n"
        f"Community Responsibility Context: {summary['community']:.1f}%\n"
        f"Largest per-applicant race-group shift signal: {top_shift:.4f}\n"
        "Do not mention policy/legal advice. Keep tone helpful and clear."
    )
    return _llm_chat_response(
        "You are a practical AI assistant that explains fairness analytics to non-technical users.",
        prompt,
        temperature=0.4,
        max_tokens=280,
    )


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _build_coach_explanation(seed: Dict[str, float], fairness: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    x_ratio = float(seed.get("xRatio", 0.5))
    y_ratio = float(seed.get("yRatio", 0.5))
    sensitivity = _normalized_sensitivity(fairness)
    shifts = fairness.get("counterfactualChanges") or []

    school_resource = (1.0 - y_ratio) * 100.0
    community_resp = y_ratio * 100.0
    family_financial = x_ratio * 100.0
    merit = (1.0 - x_ratio) * 100.0

    biggest = None
    if shifts:
        biggest = sorted(shifts, key=lambda item: item.get("maxDecisionShift", 0), reverse=True)[0]

    target_seed = target.get("suggestedSeed") or {"xRatio": x_ratio, "yRatio": y_ratio}
    target_x = float(target_seed.get("xRatio", x_ratio))
    target_y = float(target_seed.get("yRatio", y_ratio))

    lines = [
        f"Your current stance places about {school_resource:.0f}% on School Resource Context and {community_resp:.0f}% on Community Responsibility Context.",
        f"Horizontally, you are at about {merit:.0f}% Merit-Based Selection versus {family_financial:.0f}% Family Financial Context.",
        f"At this position, estimated race disparity risk score is {sensitivity:.1f} in the group-disparity audit.",
    ]
    if biggest:
        lines.append(
            f"The largest predicted race-group shift signal is Applicant {biggest.get('candidateLabel', 'Unknown')} at {float(biggest.get('maxDecisionShift', 0.0)):.3f}."
        )
    lines.append(
        f"A lower-sensitivity alternative is x={target_x:.2f}, y={target_y:.2f}, which is estimated to reduce sensitivity by {float(target.get('estimatedReduction', 0.0)):.1f} points."
    )

    action_steps = [
        "Run one admissions batch using the suggested point and compare final admits with your current point.",
        "For each borderline case, complete a race-blinded second pass before finalizing decisions.",
        "Track confidence and fairness shift together; keep changes that reduce sensitivity without collapsing evidence quality.",
    ]

    return {
        "narrative": " ".join(lines),
        "actionSteps": action_steps,
    }


def _clean_ai_text(text: str) -> str:
    return text.replace("—", ", ").replace("\u2014", ", ")


def _normalized_sensitivity(fairness: Dict[str, Any]) -> float:
    sensitivity = float(fairness.get("raceSensitivity", 0.0))
    top_pairs = fairness.get("topDisparityPairs") or []
    raw_sensitivity = fairness.get("rawRaceSensitivity")

    if isinstance(raw_sensitivity, (int, float)):
        raw_value = float(raw_sensitivity)
        if raw_value > 0.0:
            return raw_value

    if top_pairs:
        inferred_score = 100.0 * float(top_pairs[0].get("meanGap", 0.0))
        if inferred_score > 0.0:
            return inferred_score

    return sensitivity


def _sensitivity_indicator(sensitivity: float) -> str:
    if sensitivity < 1.2:
        return "unusually low bias"
    if sensitivity < 2.8:
        return "less biased than average"
    if sensitivity < 4.5:
        return "elevated bias risk"
    return "unusually high bias"


def _risk_band_label(sensitivity: float) -> str:
    if sensitivity < 1.2:
        return "Low modeled disparity risk"
    if sensitivity < 2.8:
        return "Moderate modeled disparity risk"
    if sensitivity < 4.5:
        return "Elevated modeled disparity risk"
    return "High modeled disparity risk"


def _top_pair_interpretation(top_pair: Dict[str, Any] | None, sensitivity: float) -> str:
    if not top_pair:
        return ""

    favored = str(top_pair.get("favoredRace", "tie"))
    races = top_pair.get("races") or []
    if favored == "tie" or len(races) != 2:
        return "Across applicants with similar profiles, this top comparison pair appears roughly balanced."

    race_a = str(races[0])
    race_b = str(races[1])
    other = race_b if favored == race_a else race_a

    if sensitivity >= 4.5:
        tendency = "strongly tend to prefer"
    elif sensitivity >= 2.8:
        tendency = "often prefer"
    elif sensitivity >= 1.2:
        tendency = "sometimes prefer"
    else:
        tendency = "slightly tend to prefer"

    return (
        f"A race disparity score of {sensitivity:.1f} means your modeled decisions are more likely to favor one race over another when applicants have similar profiles. "
        f"In this highest-gap comparison, you {tendency} admitting {favored} applicants over {other} applicants."
    )


def _dimension_to_value_weights(dimension_weights: Dict[str, float] | None) -> Dict[str, float]:
    if not dimension_weights:
        return {"merit": 0.25, "family": 0.25, "school": 0.25, "community": 0.25}
    return {
        "merit": float(dimension_weights.get("equity", 0.25)),
        "family": float(dimension_weights.get("individual", 0.25)),
        "school": float(dimension_weights.get("readiness", 0.25)),
        "community": float(dimension_weights.get("opportunity", 0.25)),
    }


def _value_to_dimension_weights(value_weights: Dict[str, float]) -> Dict[str, float]:
    return {
        "equity": float(value_weights.get("merit", 0.25)),
        "individual": float(value_weights.get("family", 0.25)),
        "readiness": float(value_weights.get("school", 0.25)),
        "opportunity": float(value_weights.get("community", 0.25)),
    }


def _normalize_value_weights(raw_weights: Dict[str, Any] | None) -> Dict[str, float] | None:
    if not isinstance(raw_weights, dict):
        return None
    vals = {
        "merit": max(0.0, float(raw_weights.get("merit", 0.0))),
        "family": max(0.0, float(raw_weights.get("family", 0.0))),
        "school": max(0.0, float(raw_weights.get("school", 0.0))),
        "community": max(0.0, float(raw_weights.get("community", 0.0))),
    }
    total = sum(vals.values())
    if total <= 0:
        return None
    return {k: vals[k] / total for k in vals}


def _seed_from_value_weights(value_weights: Dict[str, float]) -> Dict[str, float]:
    horizontal = float(value_weights.get("merit", 0.0)) + float(value_weights.get("family", 0.0))
    vertical = float(value_weights.get("school", 0.0)) + float(value_weights.get("community", 0.0))
    x_ratio = 0.5 if horizontal <= 0 else float(value_weights.get("family", 0.0)) / horizontal
    y_ratio = 0.5 if vertical <= 0 else float(value_weights.get("community", 0.0)) / vertical
    return {
        "xRatio": max(0.0, min(1.0, x_ratio)),
        "yRatio": max(0.0, min(1.0, y_ratio)),
    }


def _merge_analysis_state(snapshot: Dict[str, Any] | None, seed_input: Dict[str, Any] | None, value_weights_input: Dict[str, Any] | None) -> Dict[str, Any]:
    snapshot_seed = (snapshot or {}).get("seed") or {"xRatio": 0.5, "yRatio": 0.5}
    snapshot_dimension = (snapshot or {}).get("dimensionWeights") or {}
    default_weights = _dimension_to_value_weights(snapshot_dimension)

    incoming_seed_weights = None
    if isinstance(seed_input, dict):
        incoming_seed_weights = _normalize_value_weights(seed_input.get("valueWeights"))

    incoming_weights = _normalize_value_weights(value_weights_input) or incoming_seed_weights
    if incoming_weights is None and isinstance(seed_input, dict):
        x = float(seed_input.get("xRatio", snapshot_seed.get("xRatio", 0.5)))
        y = float(seed_input.get("yRatio", snapshot_seed.get("yRatio", 0.5)))
        incoming_weights = {
            "merit": max(0.0, 1.0 - x),
            "family": max(0.0, x),
            "school": max(0.0, 1.0 - y),
            "community": max(0.0, y),
        }
    if incoming_weights is None:
        incoming_weights = default_weights

    total = sum(incoming_weights.values())
    incoming_weights = {k: incoming_weights[k] / total for k in incoming_weights}
    seed = _seed_from_value_weights(incoming_weights)
    return {
        "xRatio": seed["xRatio"],
        "yRatio": seed["yRatio"],
        "valueWeights": incoming_weights,
    }


def _floor_message(sensitivity: float) -> str:
    return (
        f"Your candidate pool has inherent disparities in opportunity and preparation. "
        f"At {sensitivity:.1f}% sensitivity, this reflects the baseline gaps between racial groups in your applicants' profiles—"
        f"not primarily a result of your weighting choices. "
        f"<strong>However, you can still reduce bias by reweighting:</strong> "
        f"Moving away from merit-only selection (GPA, test scores, research) and toward contextual factors "
        f"(financial hardship, first-generation status, school resources, community impact) may help surface talent in underrepresented groups. "
        f"No single weighting will eliminate the baseline gap, but you can find positions that acknowledge disparities while making fairer decisions. "
        f"Try adjusting sliders to emphasize context over pure metrics and see how sensitivity changes."
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "version": "mvp-v2"})


@app.get("/")
def root_page():
    return redirect("/hybrid_initializer.html")


@app.get("/hybrid_initializer.html")
def hybrid_initializer_page():
    return send_from_directory(str(WEB_ROOT), "hybrid_initializer.html")


@app.get("/tradeoffGraph.html")
def tradeoff_graph_page():
    return send_from_directory(str(WEB_ROOT), "tradeoffGraph.html")


@app.get("/api/init/default-applicants")
def default_applicants():
    return jsonify({"applicants": get_default_applicants()})


@app.post("/api/init/start")
def init_start():
    body = request.get_json(silent=True) or {}
    candidates = body.get("candidateEvaluations")
    overall_rationale = str(body.get("overallRationale", "")).strip()

    if not isinstance(candidates, list) or len(candidates) < 12:
        return _json_error("candidateEvaluations must include at least twelve entries")
    if not overall_rationale:
        return _json_error("overallRationale is required")

    try:
        snapshot = build_initial_snapshot(candidates, overall_rationale)
        session_id = create_session(body, snapshot)

        fairness = compute_counterfactual_race_sensitivity(candidates, snapshot["seed"], overall_rationale)
        normalized_sensitivity = _normalized_sensitivity(fairness)
        init_explanation = generate_initialization_explanation(snapshot["dimensionWeights"], normalized_sensitivity)

        return jsonify(
            {
                "sessionId": session_id,
                "dimensionWeights": snapshot["dimensionWeights"],
                "seed": snapshot["seed"],
                "clarificationQuestions": snapshot["clarificationQuestions"],
                "featureScores": snapshot["featureScores"],
                "candidateVectors": snapshot["candidateVectors"],
                "regressionDetails": snapshot["regressionDetails"],
                "raceSensitivity": normalized_sensitivity,
                "initializationExplanation": init_explanation,
            }
        )
    except Exception as exc:
        return _json_error(f"Initialization failed: {exc}", 500)


@app.post("/api/init/answers")
def init_answers():
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    answers = body.get("answers")

    if not session_id:
        return _json_error("sessionId is required")
    if not isinstance(answers, dict) or not answers:
        return _json_error("answers must be a non-empty object")

    snapshot = get_snapshot(session_id)
    if snapshot is None:
        return _json_error("Unknown sessionId", 404)

    final_payload = finalize_weights(snapshot["dimensionWeights"], answers)
    
    # Convert dimensionWeights to valueWeights and include in seed
    final_dimension_weights = final_payload.get("dimensionWeights", {})
    final_value_weights = _dimension_to_value_weights(final_dimension_weights)
    final_payload["seed"]["valueWeights"] = final_value_weights
    
    print(f"[init_answers] Pairwise-adjusted valueWeights: {final_value_weights}", flush=True)
    
    # Compute fairness using the pairwise-adjusted value weights
    # NOT the original seed - we want the adjusted weights to affect the model
    payload = get_session_payload(session_id) or {}
    candidates = payload.get("candidateEvaluations") or []
    fairness = compute_counterfactual_race_sensitivity(
        candidates, 
        {"valueWeights": final_value_weights},  # Use adjusted value weights
        str(payload.get("overallRationale", ""))
    )
    final_payload["raceSensitivity"] = _normalized_sensitivity(fairness)
    final_payload["counterfactualChanges"] = fairness["counterfactualChanges"]
    final_payload["counterfactualPromptTemplate"] = fairness["counterfactualPromptTemplate"]

    save_answers(session_id, answers)
    save_final(session_id, final_payload)

    return jsonify(final_payload)


@app.post("/api/analysis/race-sensitivity")
def race_sensitivity():
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    seed = body.get("seed") or {}
    value_weights = body.get("valueWeights") or {}

    if not session_id:
        return _json_error("sessionId is required")

    payload = get_session_payload(session_id)
    if payload is None:
        return _json_error("Unknown sessionId", 404)

    snapshot = get_snapshot(session_id)
    merged_seed = _merge_analysis_state(snapshot, seed, value_weights)

    fairness = compute_counterfactual_race_sensitivity(
        payload.get("candidateEvaluations") or [],
        merged_seed,
        str(payload.get("overallRationale", "")),
    )
    normalized_sensitivity = _normalized_sensitivity(fairness)
    fairness["raceSensitivity"] = round(normalized_sensitivity, 2)
    fairness["disparityScore"] = round(normalized_sensitivity, 2)
    fairness["dpGap"] = round(normalized_sensitivity / 100.0, 4)
    return jsonify(fairness)


@app.post("/api/analysis/coach-insight")
def coach_insight():
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    seed = body.get("seed") or {}
    value_weights = body.get("valueWeights") or {}

    if not session_id:
        return _json_error("sessionId is required")

    payload = get_session_payload(session_id)
    if payload is None:
        return _json_error("Unknown sessionId", 404)

    snapshot = get_snapshot(session_id)
    merged_seed = _merge_analysis_state(snapshot, seed, value_weights)

    candidates = payload.get("candidateEvaluations") or []
    fairness = compute_counterfactual_race_sensitivity(candidates, merged_seed, str(payload.get("overallRationale", "")))
    target = suggest_lower_sensitivity_target(candidates, merged_seed)
    explanation = _build_coach_explanation(merged_seed, fairness, target)

    return jsonify(
        {
            "seed": merged_seed,
            "valueWeights": merged_seed.get("valueWeights", {}),
            "raceSensitivity": fairness.get("raceSensitivity", 0.0),
            "counterfactualChanges": fairness.get("counterfactualChanges", []),
            "suggestedSeed": target.get("suggestedSeed"),
            "suggestedValueWeights": target.get("suggestedValueWeights"),
            "suggestedRaceSensitivity": target.get("suggestedRaceSensitivity"),
            "estimatedReduction": target.get("estimatedReduction"),
            "narrative": explanation["narrative"],
            "actionSteps": explanation["actionSteps"],
        }
    )


@app.post("/api/analysis/dialogue")
def analysis_dialogue():
    """Returns disparity score, top race-pair gaps, and a plain-language summary."""
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    seed = body.get("seed") or {}
    value_weights = body.get("valueWeights") or {}
    
    print(f"[analysis_dialogue] Received: sessionId={session_id[:8]}..., seed keys={list(seed.keys())}, valueWeights={value_weights}", flush=True)

    if not session_id:
        return _json_error("sessionId is required")

    payload = get_session_payload(session_id)
    if payload is None:
        return _json_error("Unknown sessionId", 404)

    snapshot = get_snapshot(session_id)
    print(f"[analysis_dialogue] Snapshot seed: {snapshot.get('seed') if snapshot else 'None'}", flush=True)
    
    merged_seed = _merge_analysis_state(snapshot, seed, value_weights)
    print(f"[analysis_dialogue] Merged seed valueWeights: {merged_seed.get('valueWeights')}", flush=True)

    candidates = payload.get("candidateEvaluations") or []
    fairness = compute_counterfactual_race_sensitivity(candidates, merged_seed, str(payload.get("overallRationale", "")))
    sensitivity = _normalized_sensitivity(fairness)
    print(f"[analysis_dialogue] Computed sensitivity: {sensitivity:.1f}", flush=True)
    dp_gap = sensitivity / 100.0
    top_pairs = fairness.get("topDisparityPairs") or []
    model_diag = fairness.get("modelDiagnostics") or {}
    race_terms = model_diag.get("raceTerms") or {}

    weights = _value_to_dimension_weights(merged_seed.get("valueWeights", {}))
    summary = _build_weight_snapshot(merged_seed, weights)

    top_pair_text = "No stable race pair difference was detected in the benchmark."
    top_pair_interpretation = ""
    if top_pairs:
        top = top_pairs[0]
        pair_key = str(top.get("pairKey", "Unknown pair"))
        favored = str(top.get("favoredRace", "tie"))
        top_pair_text = (
            f"Largest average gap was {pair_key}, "
            f"with {favored} favored on average."
        )
        top_pair_interpretation = _top_pair_interpretation(top, sensitivity)

    risk_band = _risk_band_label(sensitivity)

    summary_prompt = (
        "Provide a 4-5 sentence plain-English summary of this admissions fairness result. "
        "Start with the risk band and race disparity score only. "
        "Do not use point-gap values or 'percentage points' language. "
        "Then include one explicit sentence that says: for applicants with similar profiles, which race is favored over which race in the highest-gap pair. "
        "Then mention likely value-priority drivers.\n"
        f"Risk band: {risk_band}\n"
        f"Race disparity score: {sensitivity:.1f}\n"
        f"Top pair statement: {top_pair_text}\n"
        f"Explicit interpretation sentence to include: {top_pair_interpretation or 'No explicit pairwise preference statement available.'}\n"
        f"Merit-Based Selection: {summary['merit']:.1f}%\n"
        f"Family Financial Context: {summary['family']:.1f}%\n"
        f"School Resource Context: {summary['school']:.1f}%\n"
        f"Community Responsibility Context: {summary['community']:.1f}%"
    )
    explanation = _llm_chat_response(
        "You are a careful, non-judgmental admissions analytics explainer.",
        summary_prompt,
        temperature=0.3,
        max_tokens=260,
    ) or (
        f"Risk band is {risk_band}. Race disparity score is {sensitivity:.1f}. {top_pair_text} "
        f"{top_pair_interpretation} "
        f"Your current weighting is Merit {summary['merit']:.1f}%, Family {summary['family']:.1f}%, "
        f"School {summary['school']:.1f}%, Community {summary['community']:.1f}%."
    )

    explanation = re.sub(
        r"(Race disparity score is\s*)(\d+(?:\.\d+)?)",
        f"\\g<1>{sensitivity:.1f}",
        str(explanation),
        flags=re.IGNORECASE,
    )
    explanation = re.sub(
        r"(Race disparity score:\s*)(\d+(?:\.\d+)?)",
        f"\\g<1>{sensitivity:.1f}",
        str(explanation),
        flags=re.IGNORECASE,
    )
    explanation = re.sub(
        r"(race disparity risk score:\s*)(\d+(?:\.\d+)?)",
        f"\\g<1>{sensitivity:.1f}",
        str(explanation),
        flags=re.IGNORECASE,
    )

    if top_pair_interpretation and top_pair_interpretation not in explanation:
        explanation = f"{explanation} {top_pair_interpretation}".strip()

    explanation = f"{explanation}"
    floor_reached = False

    return jsonify(
        {
            "seed": merged_seed,
            "valueWeights": merged_seed.get("valueWeights", {}),
            "sensitivity": round(sensitivity, 1),
            "sensitivityIndicator": _sensitivity_indicator(sensitivity),
            "dpGap": round(dp_gap, 4),
            "explanation": _clean_ai_text(explanation),
            "topDisparityPairs": top_pairs,
            "pairDisparities": fairness.get("pairDisparities", []),
            "floorReached": floor_reached,
            "floorMessage": "",
        }
    )


@app.post("/api/analysis/apply-suggestion")
def apply_suggestion():
    """
    applies a user-selected suggestion and logs confidence rating.
    """
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    suggestion_name = str(body.get("suggestionName", "")).strip()
    confidence_rating = int(body.get("confidenceRating", 3))
    new_seed = body.get("newSeed") or {}
    new_value_weights = body.get("newValueWeights") or {}

    if not session_id:
        return _json_error("sessionId is required")

    # Log the decision event
    log_event(
        "suggestion_applied",
        {
            "suggestionName": suggestion_name,
            "confidenceRating": confidence_rating,
            "newSeed": new_seed,
            "newValueWeights": new_value_weights,
        },
        session_id,
    )

    # Return confirmation
    return jsonify({
        "status": "logged",
        "sessionId": session_id,
        "suggestion": suggestion_name,
        "confidence": confidence_rating,
    })


@app.post("/api/analysis/ask-question")
def ask_question():
    """
    Dialogue endpoint for user questions about their stance.
    Generates AI-powered responses to user queries.
    """
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    question = str(body.get("question", "")).strip()
    seed = body.get("seed") or {}
    value_weights = body.get("valueWeights") or {}

    if not session_id or not question:
        return _json_error("sessionId and question are required")

    payload = get_session_payload(session_id)
    if payload is None:
        return _json_error("Unknown sessionId", 404)

    snapshot = get_snapshot(session_id)
    merged_seed = _merge_analysis_state(snapshot, seed, value_weights)

    options = payload.get("candidateEvaluations") or []
    fairness = compute_counterfactual_race_sensitivity(options, merged_seed, str(payload.get("overallRationale", "")))
    sens = _normalized_sensitivity(fairness)
    weights = _value_to_dimension_weights(merged_seed.get("valueWeights", {}))
    summary = _build_weight_snapshot(merged_seed, weights)
    top_pairs = fairness.get("topDisparityPairs") or []
    pair_hint = ""
    if top_pairs:
        lead = top_pairs[0]
        pair_hint = (
            f"Top disparity pair is {lead.get('pairKey', 'Unknown')} with "
            f"{100.0 * float(lead.get('meanGap', 0.0)):.1f} points."
        )

    response = _llm_question_answer(question, merged_seed, sens, fairness, weights)
    if not response:
        response = (
            f"I could not reach the language model, so here is a direct model summary. "
            f"Current race disparity score is {sens:.1f}. "
            f"{pair_hint if pair_hint else 'No stable top disparity pair was detected in this run.'} "
            f"Current weights are Merit {summary['merit']:.1f}%, Family {summary['family']:.1f}%, "
            f"School {summary['school']:.1f}%, Community {summary['community']:.1f}%. "
            f"This is a model-based risk estimate, not legal or causal proof."
        )

    return jsonify(
        {
            "response": _clean_ai_text(response),
            "regenerateSuggestions": False,
            "suggestions": [],
        }
    )


@app.post("/api/value-cards/recommend")
def value_cards_recommend():
    body = request.get_json(silent=True) or {}
    seed = body.get("seed") or {"xRatio": 0.5, "yRatio": 0.5}
    cards = recommend_value_cards(seed, limit=3)
    return jsonify({"cards": cards})


@app.post("/api/events/confidence")
def events_confidence():
    body = request.get_json(silent=True) or {}
    session_id = body.get("sessionId")
    rating = body.get("rating")

    if rating is None:
        return _json_error("rating is required")

    log_event("confidence", body, str(session_id) if session_id else None)
    return jsonify({"status": "logged"})


@app.post("/api/events/reflection")
def events_reflection():
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("sessionId", "")).strip()
    responses = body.get("responses") or {}

    if not isinstance(responses, dict):
        return _json_error("responses must be an object")

    safe_session_id = session_id or "anonymous"

    log_event(
        "reflection",
        {
            "responses": responses,
            "source": "tradeoffGraph",
        },
        safe_session_id,
    )
    table_path = append_reflection_submission(safe_session_id, responses)
    return jsonify({"status": "logged", "tablePath": str(table_path)})


@app.get("/api/export/sessions.csv")
def export_sessions_csv():
    rows = export_sessions()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["session_id", "created_at", "payload_json", "snapshot_json", "final_json"])
    for row in rows:
        writer.writerow(
            [
                row.get("session_id", ""),
                row.get("created_at", ""),
                row.get("payload_json", ""),
                row.get("snapshot_json", ""),
                row.get("final_json", ""),
            ]
        )
    csv_content = output.getvalue()
    return Response(csv_content, mimetype="text/csv")


@app.get("/api/export/reflections.csv")
def export_reflections_csv():
    rows = export_reflections()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "event_id",
            "session_id",
            "created_at",
            "surprise_level",
            "seen_similar_tool_before",
            "usefulness_rating",
            "recommend_rating",
            "change_intent_rating",
            "reflection_text",
            "submitted_at",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.get("event_id", ""),
                row.get("session_id", ""),
                row.get("created_at", ""),
                row.get("surprise_level", ""),
                row.get("seen_similar_tool_before", ""),
                row.get("usefulness_rating", ""),
                row.get("recommend_rating", ""),
                row.get("change_intent_rating", ""),
                row.get("reflection_text", ""),
                row.get("submitted_at", ""),
            ]
        )

    csv_content = output.getvalue()
    return Response(csv_content, mimetype="text/csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
