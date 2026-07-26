"""Combine fatigue, fall, and gesture cues into one explainable distress score."""

from __future__ import annotations

from typing import Any


class DistressEngine:
    """Aggregate multiple signals into a single risk score and level."""

    def __init__(self) -> None:
        self.last_reason: str | None = None

    def evaluate(
        self,
        fatigue_risk: float | None = None,
        fall_result: dict[str, Any] | None = None,
        gesture_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine cues into a normalized risk score, confidence, reason, and level."""
        fatigue_value = 0.0 if fatigue_risk is None else float(fatigue_risk)
        fall_value = 0.0
        gesture_value = 0.0
        reasons: list[str] = []

        if fall_result:
            status = str(fall_result.get("status", "Unknown")).lower()
            if status == "fall":
                fall_value = 1.0
                reasons.append("fall detected")
            elif status == "lying down":
                fall_value = 0.75
                reasons.append("person is lying down")
            elif status == "sitting":
                fall_value = 0.25
            elif status == "standing":
                fall_value = 0.0

            if bool(fall_result.get("alert", False)):
                reasons.append("alert is active")

        if gesture_result:
            gesture = str(gesture_result.get("gesture", "Unknown"))
            if gesture == "Open Palm":
                gesture_value = 0.7 if not gesture_result.get("distress_signal", False) else 0.95
                reasons.append("open palm gesture")
            elif gesture == "Raised Hand":
                gesture_value = 0.45
                reasons.append("raised hand")
            elif gesture == "Closed Fist":
                gesture_value = 0.2
                reasons.append("closed fist")

            if bool(gesture_result.get("distress_signal", False)):
                reasons.append("possible distress signal")

        combined_score = min(1.0, max(0.0, 0.55 * (fatigue_value / 100.0) + 0.3 * fall_value + 0.15 * gesture_value))
        confidence = min(1.0, 0.5 + 0.3 * (fatigue_value / 100.0) + 0.1 * fall_value + 0.1 * gesture_value)

        if combined_score < 0.2:
            level = "Normal"
        elif combined_score < 0.4:
            level = "Low"
        elif combined_score < 0.7:
            level = "Medium"
        elif combined_score < 0.85:
            level = "High"
        else:
            level = "Critical"

        risk_score = round(combined_score * 100.0, 1)
        confidence_score = round(confidence * 100.0, 1)
        reason_text = "; ".join(reasons) if reasons else "No strong distress cues detected"
        self.last_reason = reason_text

        return {
            "risk_score": risk_score,
            "confidence": confidence_score,
            "reason": reason_text,
            "level": level,
        }
