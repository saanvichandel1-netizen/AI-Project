"""MediaPipe Pose-based activity detector for standing, sitting, and walking."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment fallback
    mp = None


class MediaPipePoseDetector:
    """Wrap MediaPipe Pose inference and produce posture labels and overlays."""

    def __init__(self) -> None:
        self.pose = None
        self.mp_drawing = None
        self.mp_pose = None

        if mp is None:
            return

        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
        except Exception:
            self.pose = None
            self.mp_drawing = None
            self.mp_pose = None

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, str, Optional[list[dict[str, object]]]]:
        """Return the annotated frame, activity label, and landmark list."""
        annotated = frame.copy()
        landmarks: list[dict[str, object]] = []
        label = "Uncertain"

        if self.pose is None or self.mp_drawing is None or self.mp_pose is None:
            cv2.putText(annotated, "Pose unavailable", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return annotated, label, landmarks

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )

            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(landmark.visibility or 0.0),
                })

            label = self._classify_activity(results.pose_landmarks)
            cv2.putText(
                annotated,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return annotated, label, landmarks

    def _classify_activity(self, pose_landmarks) -> str:
        """Heuristically classify the pose as standing, sitting, or walking."""
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_leg_angle = self._angle(left_hip, left_knee, left_ankle)
        right_leg_angle = self._angle(right_hip, right_knee, right_ankle)
        avg_leg_angle = (left_leg_angle + right_leg_angle) / 2.0

        if avg_leg_angle < 100:
            return "Sitting"
        if avg_leg_angle > 150:
            return "Walking"
        return "Standing"

    @staticmethod
    def _angle(a, b, c) -> float:
        """Compute an angle in degrees between three landmarks."""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float32)
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float32)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
