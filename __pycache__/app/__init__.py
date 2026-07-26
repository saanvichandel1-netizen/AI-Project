"""Application package for the HelpGuard AI surveillance demo."""

from .detection import SafetyStateProcessor
from .metrics import (
    compute_eye_aperture,
    compute_fatigue_risk_score,
    compute_mouth_aspect_ratio,
    evaluate_ocular_metrics,
    log_incident,
)

__all__ = [
    "SafetyStateProcessor",
    "compute_eye_aperture",
    "compute_fatigue_risk_score",
    "compute_mouth_aspect_ratio",
    "evaluate_ocular_metrics",
    "log_incident",
]
