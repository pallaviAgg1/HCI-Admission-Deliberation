from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List


FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "gpa": ["gpa", "grade point average", "grades", "academic record"],
    "test_scores": ["test", "sat", "act", "scores", "exam"],
    "research": ["research", "publication", "lab", "project"],
    "leadership": ["leadership", "captain", "led", "initiative"],
    "first_gen": ["first-gen", "first generation", "first gen", "family college"],
    "adversity": ["adversity", "hardship", "obstacle", "challenge", "resilience"],
    "community_impact": ["community", "service", "volunteer", "impact", "mentor"],
    "gender_equity": ["gender", "women", "men", "nonbinary", "equity", "representation"],
    "efficiency": ["efficiency", "fast", "quick", "scalable", "throughput", "time"],
}

POSITIVE_TOKENS = {
    "important",
    "strong",
    "excellent",
    "outstanding",
    "high",
    "impressive",
    "value",
    "prioritize",
    "prioritise",
    "admit",
    "plus",
}

NEGATIVE_TOKENS = {
    "weak",
    "low",
    "poor",
    "concerning",
    "risk",
    "worry",
    "reject",
    "minus",
    "less",
    "not",
}


@dataclass
class FeatureSignal:
    present: int
    score: float
    evidence: List[str]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\-']+", text.lower())


def _window_sentiment(tokens: List[str], index: int, window: int = 4) -> float:
    left = max(0, index - window)
    right = min(len(tokens), index + window + 1)
    nearby = tokens[left:right]
    pos_hits = sum(1 for token in nearby if token in POSITIVE_TOKENS)
    neg_hits = sum(1 for token in nearby if token in NEGATIVE_TOKENS)
    if pos_hits == 0 and neg_hits == 0:
        return 0.2
    return float(pos_hits - neg_hits)


def extract_features(text: str) -> Dict[str, FeatureSignal]:
    """Rule-based baseline extractor with a stable contract for model replacement later."""
    tokens = _tokenize(text)
    joined = " ".join(tokens)

    result: Dict[str, FeatureSignal] = {}
    for feature, patterns in FEATURE_KEYWORDS.items():
        evidence: List[str] = []
        score = 0.0
        for pattern in patterns:
            if pattern in joined:
                evidence.append(pattern)
                parts = pattern.split()
                for idx in range(len(tokens) - len(parts) + 1):
                    if tokens[idx : idx + len(parts)] == parts:
                        score += _window_sentiment(tokens, idx)
        present = 1 if evidence else 0
        if present and score == 0.0:
            score = 0.2
        result[feature] = FeatureSignal(present=present, score=score, evidence=evidence)

    return result
