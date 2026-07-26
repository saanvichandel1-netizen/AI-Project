"""Explainability helpers for distress alerts."""

from __future__ import annotations

from typing import Any

import numpy as np


class XAIExplainer:
    """Provide textual explanations and optional heatmap-style overlays."""

    def __init__(self) -> None:
        self.grad_cam_available = False

    def explain(self, distress_result: dict[str, Any], fatigue_risk: float | None = None, fall_result: dict[str, Any] | None = None, gesture_result: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return explanation details for the dashboard."""
        features: list[str] = []
        reasons: list[str] = []

        if fatigue_risk is not None and fatigue_risk > 60:
            features.append("Low Eye Aspect Ratio")
            reasons.append("The facial metrics suggest fatigue or drowsiness")

        if fall_result and str(fall_result.get("status", "")).lower() in {"fall", "lying down"}:
            features.append("Fall Detected")
            reasons.append("The pose layout indicates a fall or prolonged lying posture")

        if gesture_result:
            gesture = str(gesture_result.get("gesture", "Unknown"))
            if gesture == "Open Palm" and bool(gesture_result.get("distress_signal", False)):
                features.append("Raised Hand")
                reasons.append("A sustained open palm was interpreted as a possible distress cue")
            elif gesture == "Raised Hand":
                features.append("Raised Hand")
            elif gesture == "Closed Fist":
                features.append("Closed Fist")

        if not features:
            features.append("No strong cues")
            reasons.append("The current scene has not triggered a strong distress signal")

        explanation = {
            "reason": " | ".join(reasons),
            "confidence": distress_result.get("confidence", 0.0),
            "detected_features": features,
            "risk_score": distress_result.get("risk_score", 0.0),
            "heatmap": None,
        }

        return explanation
