from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


_MODEL = None
_MODEL_ERROR: str | None = None
_PROTOTYPE_CACHE: Dict[str, np.ndarray] = {}


TRAIT_PROTOTYPES: Dict[str, Dict[str, List[str]]] = {
    "research": {
        "high": [
            "I designed experiments, analyzed data, and documented findings with a faculty mentor.",
            "My work involved lab protocols, iterative testing, and publishing or presenting results.",
        ],
        "low": [
            "I have limited exposure to formal research projects or laboratory work.",
            "Most of my activities were not research-oriented and did not involve experiments.",
        ],
    },
    "leadership": {
        "high": [
            "I organized teams, led initiatives, and coordinated people toward shared goals.",
            "I founded programs, mentored peers, and took responsibility for outcomes.",
        ],
        "low": [
            "I mostly participated individually and did not take sustained leadership roles.",
            "My involvement was limited to membership without organizing responsibilities.",
        ],
    },
    "adversity": {
        "high": [
            "I balanced school with caregiving, financial pressure, and structural barriers to opportunity.",
            "Limited resources, work obligations, and family responsibilities shaped my academic path.",
        ],
        "low": [
            "I experienced relatively stable access to academic resources and support.",
            "My schooling faced few external constraints from finances, caregiving, or access barriers.",
        ],
    },
    "community_impact": {
        "high": [
            "I led sustained service projects that produced measurable benefits for my community.",
            "I coordinated volunteers and delivered recurring mentorship, outreach, or advocacy work.",
        ],
        "low": [
            "I had minimal involvement in community-facing service or outreach activities.",
            "My activities were mostly individual and not centered on community impact.",
        ],
    },
    "first_gen": {
        "high": [
            "I am a first-generation college student and navigated admissions without family college experience.",
            "As the first in my family to pursue college, I translated systems for my household.",
        ],
        "low": [
            "My family already has college experience and I received direct college guidance at home.",
            "I am not first-generation and had family familiarity with higher education pathways.",
        ],
    },
}


def _fallback_scores(profile: Dict[str, Any]) -> Dict[str, float]:
    return {
        "research": _clamp(float(profile.get("research", 0.0)) / 10.0),
        "leadership": _clamp(float(profile.get("leadership", 0.0)) / 10.0),
        "adversity": _clamp(float(profile.get("adversity", 0.0)) / 10.0),
        "community_impact": _clamp(float(profile.get("communityImpact", 0.0)) / 10.0),
        "first_gen": 1.0 if bool(profile.get("firstGen", False)) else 0.0,
        "review_complexity": _clamp(float(profile.get("reviewComplexity", 5.0)) / 10.0),
    }


def _load_model() -> Any | None:
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        return None
    if SentenceTransformer is None:
        _MODEL_ERROR = "sentence-transformers not installed"
        return None

    model_name = os.getenv("SEMANTIC_PROFILE_MODEL", "all-MiniLM-L6-v2")
    try:
        _MODEL = SentenceTransformer(model_name)
        return _MODEL
    except Exception as exc:
        _MODEL_ERROR = str(exc)
        return None


def _prototype_vector(model: Any, trait: str, bucket: str) -> np.ndarray:
    key = f"{trait}:{bucket}"
    cached = _PROTOTYPE_CACHE.get(key)
    if cached is not None:
        return cached

    texts = TRAIT_PROTOTYPES[trait][bucket]
    vectors = model.encode(texts, normalize_embeddings=True)
    proto = np.mean(np.array(vectors, dtype=float), axis=0)
    norm = np.linalg.norm(proto)
    if norm > 0:
        proto = proto / norm
    _PROTOTYPE_CACHE[key] = proto
    return proto


def _semantic_trait_score(model: Any, text_vec: np.ndarray, trait: str) -> float:
    high = _prototype_vector(model, trait, "high")
    low = _prototype_vector(model, trait, "low")
    sim_high = float(np.dot(text_vec, high))
    sim_low = float(np.dot(text_vec, low))
    # Map [-1, 1] style margin to [0, 1]
    return _clamp(0.5 + 0.5 * (sim_high - sim_low))


def infer_semantic_profile_scores(candidate: Dict[str, Any]) -> Dict[str, float]:
    profile = candidate.get("profile") or {}
    fallback = _fallback_scores(profile)

    text_parts = [
        str(candidate.get("personalStatement", "") or "").strip(),
        str(candidate.get("resumeText", "") or "").strip(),
        str(candidate.get("resume", "") or "").strip(),
        str(candidate.get("summary", "") or "").strip(),
    ]
    text = "\n".join(part for part in text_parts if part)

    if len(text.split()) < 20:
        return fallback

    model = _load_model()
    if model is None:
        return fallback

    try:
        vec = model.encode([text], normalize_embeddings=True)
        text_vec = np.array(vec[0], dtype=float)

        research = _semantic_trait_score(model, text_vec, "research")
        leadership = _semantic_trait_score(model, text_vec, "leadership")
        adversity = _semantic_trait_score(model, text_vec, "adversity")
        community_impact = _semantic_trait_score(model, text_vec, "community_impact")
        first_gen = _semantic_trait_score(model, text_vec, "first_gen")

        # Keep a light anchor to existing structured data for stability.
        mix = 0.85
        research = mix * research + (1.0 - mix) * fallback["research"]
        leadership = mix * leadership + (1.0 - mix) * fallback["leadership"]
        adversity = mix * adversity + (1.0 - mix) * fallback["adversity"]
        community_impact = mix * community_impact + (1.0 - mix) * fallback["community_impact"]
        first_gen = mix * first_gen + (1.0 - mix) * fallback["first_gen"]

        review_complexity = _clamp(
            0.40 * adversity + 0.25 * leadership + 0.20 * community_impact + 0.15 * research
        )

        return {
            "research": _clamp(research),
            "leadership": _clamp(leadership),
            "adversity": _clamp(adversity),
            "community_impact": _clamp(community_impact),
            "first_gen": _clamp(first_gen),
            "review_complexity": review_complexity,
        }
    except Exception:
        return fallback
