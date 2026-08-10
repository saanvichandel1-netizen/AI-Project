"""Fatigue safety processor built on MediaPipe Face Mesh."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .metrics import (
    build_explanation,
    compute_eye_aperture,
    compute_mouth_aspect_ratio,
    evaluate_ocular_metrics,
    get_mediapipe_app,
    plot_text,
)


class SafetyStateProcessor:
    """Process frames to detect fatigue indicators and annotate the video."""

    def __init__(self, face_mesh: Optional[Any] = None) -> None:
        self.face_mesh = face_mesh or get_mediapipe_app()

    def process(self, frame: np.ndarray, *, ear_threshold: float = 0.25, mar_threshold: float = 0.45) -> Dict[str, Any]:
        """Analyze one frame and return the resulting metrics plus an annotated frame."""
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        metrics = {
            "ear": 0.0,
            "mar": 0.0,
            "risk_score": 0.0,
            "status": "Monitoring",
            "reason": "Normal monitoring.",
        }
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear = compute_eye_aperture(landmarks, width, height, left_eye=True)
            mar = compute_mouth_aspect_ratio(landmarks, width, height)
            metrics.update(evaluate_ocular_metrics(ear, mar, ear_threshold=ear_threshold, mar_threshold=mar_threshold))
            metrics["reason"] = build_explanation(ear, mar, ear_threshold=ear_threshold, mar_threshold=mar_threshold)

            color = (0, 0, 255) if metrics["status"] == "Alert" else (0, 255, 0)
            cv2.rectangle(frame, (10, 10), (width - 10, 95), color, 2)
            plot_text(frame, f"EAR: {metrics['ear']:.3f}", (20, 35), color)
            plot_text(frame, f"MAR: {metrics['mar']:.3f}", (20, 60), color)
            plot_text(frame, f"Status: {metrics['status']}", (20, 85), color)

            cv2.putText(
                frame,
                "HelpGuard AI",
                (width - 180, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return {"frame": frame, **metrics}
