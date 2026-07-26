"""Pose-based fall and posture detector for standing, sitting, lying down, and falls."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment fallback
    mp = None


class MediaPipeFallDetector:
    """Detect posture states from MediaPipe Pose landmarks and raise alerts for prolonged lying."""

    def __init__(self) -> None:
        self.pose = None
        self.mp_pose = None
        self.mp_drawing = None
        self.last_center_y: float | None = None
        self.last_timestamp: float | None = None
        self.lie_start_time: float | None = None

        if mp is None:
            return

        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception:
            self.pose = None
            self.mp_pose = None
            self.mp_drawing = None

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Return posture classification, confidence, reason, and alert state."""
        annotated = frame.copy()
        if self.pose is None or self.mp_pose is None or self.mp_drawing is None:
            return {
                "status": "Unknown",
                "confidence": 0.0,
                "reason": "Pose inference unavailable",
                "alert": False,
            }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return {
                "status": "Unknown",
                "confidence": 0.0,
                "reason": "No pose landmarks detected",
                "alert": False,
            }

        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

        landmarks = results.pose_landmarks.landmark
        frame_h, frame_w = frame.shape[:2]

        shoulder_mid = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], frame_w, frame_h)
        shoulder_right = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value], frame_w, frame_h)
        hip_left = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], frame_w, frame_h)
        hip_right = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value], frame_w, frame_h)
        knee_left = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value], frame_w, frame_h)
        knee_right = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value], frame_w, frame_h)
        ankle_left = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value], frame_w, frame_h)
        ankle_right = self._pixel_point(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value], frame_w, frame_h)

        mid_shoulder = ((shoulder_mid[0] + shoulder_right[0]) / 2.0, (shoulder_mid[1] + shoulder_right[1]) / 2.0)
        mid_hip = ((hip_left[0] + hip_right[0]) / 2.0, (hip_left[1] + hip_right[1]) / 2.0)
        mid_ankle = ((ankle_left[0] + ankle_right[0]) / 2.0, (ankle_left[1] + ankle_right[1]) / 2.0)

        torso_vector = (mid_shoulder[0] - mid_hip[0], mid_shoulder[1] - mid_hip[1])
        torso_angle = self._angle_to_vertical(torso_vector)
        leg_angle = self._mean_angle(hip_left, knee_left, ankle_left, hip_right, knee_right, ankle_right)
        body_height = self._distance(mid_shoulder, mid_ankle)
        center_y = (mid_shoulder[1] + mid_hip[1] + mid_ankle[1]) / 3.0

        current_time = time.perf_counter()
        delta_time = 0.0
        if self.last_timestamp is not None:
            delta_time = current_time - self.last_timestamp

        self.last_timestamp = current_time

        if torso_angle > 70 and body_height < 0.5 * max(frame_h, frame_w):
            if self.last_center_y is not None and (center_y - self.last_center_y) > 0.12 and delta_time < 1.0:
                self.last_center_y = center_y
                return {
                    "status": "Fall",
                    "confidence": 0.92,
                    "reason": "Body dropped sharply and is now horizontally aligned",
                    "alert": True,
                    "frame": annotated,
                }

            if self.lie_start_time is None:
                self.lie_start_time = current_time
            lie_duration = current_time - self.lie_start_time
            if lie_duration >= 3.0:
                self.last_center_y = center_y
                return {
                    "status": "Lying Down",
                    "confidence": 0.9,
                    "reason": f"Person has remained horizontal for {lie_duration:.1f}s",
                    "alert": True,
                    "frame": annotated,
                }

            self.last_center_y = center_y
            return {
                "status": "Lying Down",
                "confidence": 0.83,
                "reason": "Body is near horizontal and low to the ground",
                "alert": False,
                "frame": annotated,
            }

        if torso_angle < 35 and leg_angle > 150:
            self.lie_start_time = None
            self.last_center_y = center_y
            return {
                "status": "Standing",
                "confidence": 0.88,
                "reason": "Torso is upright and knees are extended",
                "alert": False,
                "frame": annotated,
            }

        if torso_angle < 45 and leg_angle < 140:
            self.lie_start_time = None
            self.last_center_y = center_y
            return {
                "status": "Sitting",
                "confidence": 0.84,
                "reason": "Torso is upright but knees are bent",
                "alert": False,
                "frame": annotated,
            }

        self.lie_start_time = None
        self.last_center_y = center_y
        return {
            "status": "Unknown",
            "confidence": 0.55,
            "reason": "Bending or an ambiguous pose was detected and ignored",
            "alert": False,
            "frame": annotated,
        }

    @staticmethod
    def _pixel_point(landmark: Any, frame_w: int, frame_h: int) -> tuple[float, float]:
        """Convert normalized MediaPipe coordinates to pixel space."""
        x = int(round(landmark.x * frame_w))
        y = int(round(landmark.y * frame_h))
        return float(x), float(y)

    @staticmethod
    def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    @staticmethod
    def _angle_to_vertical(vector: tuple[float, float]) -> float:
        if np.linalg.norm(vector) < 1e-6:
            return 90.0
        angle = np.degrees(np.arctan2(abs(vector[0]), -vector[1]))
        return float(angle)

    @staticmethod
    def _mean_angle(
        hip_left: tuple[float, float],
        knee_left: tuple[float, float],
        ankle_left: tuple[float, float],
        hip_right: tuple[float, float],
        knee_right: tuple[float, float],
        ankle_right: tuple[float, float],
    ) -> float:
        left_angle = MediaPipeFallDetector._angle(hip_left, knee_left, ankle_left)
        right_angle = MediaPipeFallDetector._angle(hip_right, knee_right, ankle_right)
        return float((left_angle + right_angle) / 2.0)

    @staticmethod
    def _angle(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
        ba = np.array(point_a) - np.array(point_b)
        bc = np.array(point_c) - np.array(point_b)
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
