"""Backward-compatible re-export for the refactored detection module."""

from app.detection import SafetyStateProcessor
from app.metrics import (
    compute_eye_aperture,
    compute_fatigue_risk_score,
    compute_mouth_aspect_ratio,
    evaluate_ocular_metrics,
    get_mediapipe_app,
    log_incident,
    plot_eye_landmarks,
    plot_text,
)

__all__ = [
    "SafetyStateProcessor",
    "compute_eye_aperture",
    "compute_fatigue_risk_score",
    "compute_mouth_aspect_ratio",
    "evaluate_ocular_metrics",
    "get_mediapipe_app",
    "log_incident",
    "plot_eye_landmarks",
    "plot_text",
]
