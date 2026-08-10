"""Utility functions for fatigue-related facial metric analysis."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _get_landmark_point(landmarks: List[object], index: int, width: int, height: int) -> Tuple[float, float]:
    """Convert a MediaPipe landmark index into pixel coordinates."""
    landmark = landmarks[index]
    return int(landmark.x * width), int(landmark.y * height)


def compute_eye_aperture(landmarks: List[object], width: int, height: int, *, left_eye: bool = True) -> float:
    """Compute the Eye Aspect Ratio (EAR) from the selected eye landmarks."""
    if left_eye:
        points = [159, 145, 386, 374, 385, 381]
    else:
        points = [33, 160, 387, 373, 386, 380]

    coords = [_get_landmark_point(landmarks, idx, width, height) for idx in points]
    p1, p2, p3, p4, p5, p6 = coords

    vertical_1 = np.linalg.norm(np.array(p1) - np.array(p2))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p4))
    horizontal = np.linalg.norm(np.array(p5) - np.array(p6))

    if horizontal <= 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mouth_aspect_ratio(landmarks: List[object], width: int, height: int) -> float:
    """Compute the Mouth Aspect Ratio (MAR) using the lip landmarks."""
    points = [61, 185, 40, 39, 0, 17]
    coords = [_get_landmark_point(landmarks, idx, width, height) for idx in points]
    p1, p2, p3, p4, p5, p6 = coords

    vertical = np.linalg.norm(np.array(p1) - np.array(p2))
    horizontal = np.linalg.norm(np.array(p3) - np.array(p4))
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


def compute_fatigue_risk_score(ear: float, mar: float, ear_threshold: float = 0.25, mar_threshold: float = 0.45) -> float:
    """Convert facial metrics into a simple fatigue risk score between 0 and 1."""
    ear_component = max(0.0, (ear_threshold - ear) / ear_threshold)
    mar_component = max(0.0, (mar - mar_threshold) / max(0.01, mar_threshold))
    return round(min(1.0, 0.6 * ear_component + 0.4 * mar_component), 3)


def evaluate_ocular_metrics(ear: float, mar: float, *, ear_threshold: float = 0.25, mar_threshold: float = 0.45) -> Dict[str, object]:
    """Return a summary dictionary for the current frame."""
    risk_score = compute_fatigue_risk_score(ear, mar, ear_threshold, mar_threshold)
    status = "Alert" if ear < ear_threshold or mar > mar_threshold else "Monitoring"
    return {
        "ear": round(ear, 3),
        "mar": round(mar, 3),
        "risk_score": risk_score,
        "status": status,
    }


def build_explanation(ear: float, mar: float, ear_threshold: float = 0.25, mar_threshold: float = 0.45) -> str:
    """Generate a simple explanation for why the current frame triggered an alert."""
    if ear < ear_threshold and mar > mar_threshold:
        return "Alert: eyes remained closed below threshold and yawning was detected."
    if ear < ear_threshold:
        return "Alert: eyes remained closed below threshold."
    if mar > mar_threshold:
        return "Alert: yawning detected."
    return "Normal monitoring."


def get_mediapipe_app() -> mp.solutions.face_mesh.FaceMesh:
    """Create the MediaPipe Face Mesh detector used for landmark extraction."""
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def plot_text(frame: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a text label over the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def log_incident(event: str, metrics: Dict[str, object]) -> Dict[str, object]:
    """Return a structured incident record for logging and reporting."""
    return {
        "event": event,
        "ear": metrics.get("ear"),
        "mar": metrics.get("mar"),
        "risk_score": metrics.get("risk_score"),
        "status": metrics.get("status"),
    }
