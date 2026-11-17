"""Transparent formulas for converting visual evidence into a health score."""

from __future__ import annotations


def condition_score(damage_fraction: float, confidence: float) -> float:
    """Return a 0-100 score where 100 means no visible damage."""
    bounded_fraction = min(max(damage_fraction, 0.0), 1.0)
    bounded_confidence = min(max(confidence, 0.0), 1.0)
    penalty = 100.0 * bounded_fraction * bounded_confidence
    return round(max(0.0, 100.0 - penalty), 2)


def priority(score: float, criticality: float = 1.0) -> str:
    """Classify intervention urgency using a transparent severity rule."""
    weighted_damage = (100.0 - min(max(score, 0.0), 100.0)) * max(criticality, 0.0)
    if weighted_damage >= 60:
        return "urgent"
    if weighted_damage >= 30:
        return "planned"
    return "monitor"
