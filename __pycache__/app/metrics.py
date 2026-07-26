"""Shared image-processing helpers and explainable surveillance metric calculations."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment fallback
    mp = None


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int]:
    """Convert normalized coordinates to pixel coordinates for older MediaPipe versions."""
    x = int(round(normalized_x * image_width))
    y = int(round(normalized_y * image_height))
    return max(0, min(x, max(image_width - 1, 0))), max(0, min(y, max(image_height - 1, 0)))


denormalize_coordinates = _normalized_to_pixel_coordinates


class _FallbackFaceMesh:
    """Fallback object used when MediaPipe face-mesh modules are unavailable."""

    def process(self, _image: np.ndarray) -> Any:
        return type("FaceMeshResult", (), {"multi_face_landmarks": None})()


def get_mediapipe_app(
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.solutions.face_mesh.FaceMesh:
    """Initialize and return a MediaPipe FaceMesh solution instance."""
    if mp is None:
        return _FallbackFaceMesh()

    try:
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    except Exception:
        return _FallbackFaceMesh()


def distance(point_1: Sequence[float], point_2: Sequence[float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return float(sum((i - j) ** 2 for i, j in zip(point_1, point_2)) ** 0.5)


def compute_eye_aperture(
    landmarks: Sequence[Any],
    refer_idxs: Sequence[int],
    frame_width: int,
    frame_height: int,
) -> Tuple[float, Optional[list[Tuple[int, int]]]]:
    """Calculate the Eye Aspect Ratio for one eye."""
    try:
        coords_points: list[Tuple[int, int]] = []
        for index in refer_idxs:
            landmark = landmarks[index]
            coord = denormalize_coordinates(landmark.x, landmark.y, frame_width, frame_height)
            coords_points.append(coord)

        p2_p6 = distance(coords_points[1], coords_points[5])
        p3_p5 = distance(coords_points[2], coords_points[4])
        p1_p4 = distance(coords_points[0], coords_points[3])

        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear, coords_points
    except (AttributeError, IndexError, TypeError):
        return 0.0, None


def compute_mouth_aspect_ratio(
    landmarks: Sequence[Any],
    refer_idxs: Sequence[int],
    frame_width: int,
    frame_height: int,
) -> float:
    """Calculate the Mouth Aspect Ratio for the chosen mouth landmarks."""
    coords = [
        denormalize_coordinates(landmarks[index].x, landmarks[index].y, frame_width, frame_height)
        for index in refer_idxs
    ]

    vertical_distance = distance(coords[2], coords[3])
    horizontal_distance = distance(coords[0], coords[1])

    return vertical_distance / horizontal_distance if horizontal_distance != 0 else 0.0


def evaluate_ocular_metrics(
    landmarks: Sequence[Any],
    left_eye_idxs: Sequence[int],
    right_eye_idxs: Sequence[int],
    image_w: int,
    image_h: int,
) -> Tuple[float, Tuple[Optional[list[Tuple[int, int]]], Optional[list[Tuple[int, int]]]]]:
    """Compute eye metrics for the left and right eyes."""
    left_ear, left_lm_coordinates = compute_eye_aperture(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = compute_eye_aperture(landmarks, right_eye_idxs, image_w, image_h)
    average_ear = (left_ear + right_ear) / 2.0

    return average_ear, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(
    frame: np.ndarray,
    left_lm_coordinates: Optional[Sequence[Tuple[int, int]]],
    right_lm_coordinates: Optional[Sequence[Tuple[int, int]]],
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw detected eye landmarks onto the frame and return the flipped frame."""
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    return cv2.flip(frame, 1)


def plot_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    fnt_scale: float = 0.8,
    thickness: int = 2,
) -> np.ndarray:
    """Render text on an image and return the updated frame."""
    return cv2.putText(image, text, origin, font, fnt_scale, color, thickness)


def compute_fatigue_risk_score(ear: float, mar: float, weight_eye: float = 0.7, weight_mouth: float = 0.3) -> float:
    """Combine eye and mouth cues into a single distress risk score between 0 and 100."""
    eye_component = max(0.0, 1.0 - ear)
    mouth_component = max(0.0, mar)
    score = (weight_eye * eye_component) + (weight_mouth * mouth_component)
    return min(1.0, max(0.0, score)) * 100.0


def log_incident(event_type: str, duration_seconds: float, log_path: str = "incident_log.csv") -> None:
    """Persist an alert event to a local CSV file for later analysis."""
    from .persistence import append_incident

    append_incident(event_type, duration_seconds, log_path=log_path)
