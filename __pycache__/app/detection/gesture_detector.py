"""MediaPipe Hands-based gesture detector for raised hand, open palm, and closed fist."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment fallback
    mp = None


class MediaPipeGestureDetector:
    """Recognize hand gestures and surface distress-like signals for prolonged open palms."""

    def __init__(self) -> None:
        self.hands = None
        self.mp_hands = None
        self.mp_drawing = None
        self.open_palm_start_time: float | None = None

        if mp is None:
            return

        try:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception:
            self.hands = None
            self.mp_hands = None
            self.mp_drawing = None

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Return gesture, confidence, reason, and whether a distress signal is suggested."""
        annotated = frame.copy()
        if self.hands is None or self.mp_hands is None or self.mp_drawing is None:
            return {
                "gesture": "Unknown",
                "confidence": 0.0,
                "reason": "Hands inference unavailable",
                "distress_signal": False,
            }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self.open_palm_start_time = None
            return {
                "gesture": "Unknown",
                "confidence": 0.0,
                "reason": "No hands detected",
                "distress_signal": False,
            }

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )

        gesture, confidence, reason = self._classify_hands(results.multi_hand_landmarks)
        distress_signal = False

        if gesture == "Open Palm":
            if self.open_palm_start_time is None:
                self.open_palm_start_time = time.perf_counter()
            elapsed = time.perf_counter() - self.open_palm_start_time
            if elapsed >= 2.0:
                distress_signal = True
                reason = f"Open palm visible for {elapsed:.1f}s — possible distress signal"
        else:
            self.open_palm_start_time = None

        return {
            "gesture": gesture,
            "confidence": confidence,
            "reason": reason,
            "distress_signal": distress_signal,
            "frame": annotated,
        }

    def _classify_hands(self, hand_landmarks: list[Any]) -> tuple[str, float, str]:
        """Heuristically classify the visible hand pose."""
        if not hand_landmarks:
            return "Unknown", 0.0, "No hands detected"

        landmarks = hand_landmarks[0].landmark
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        fingertip_y = [tip.y for tip in finger_tips]
        palm_open = max(fingertip_y) - min(fingertip_y)
        hand_height = abs(wrist.y - max(fingertip_y))

        if palm_open > 0.08 and hand_height > 0.02:
            return "Open Palm", 0.9, "All fingers are spread and the palm is exposed"

        if max(fingertip_y) < wrist.y - 0.02:
            return "Closed Fist", 0.86, "Fingers are curled inward and the hand is compact"

        return "Raised Hand", 0.82, "The hand is elevated with fingers extended"
