"""Detection pipeline for explainable surveillance distress monitoring."""

from __future__ import annotations

import time
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from .config import DEFAULT_THRESHOLDS, EYE_INDICES, MOUTH_INDICES
from .models import DetectionThresholds
from .metrics import (
    compute_fatigue_risk_score,
    compute_mouth_aspect_ratio,
    evaluate_ocular_metrics,
    get_mediapipe_app,
    plot_eye_landmarks,
    plot_text,
    log_incident,
)


class SafetyStateProcessor:
    """Track fatigue indicators from facial landmarks and emit explainable alerts."""

    def __init__(self) -> None:
        self.eye_idxs = EYE_INDICES
        self.mouth_idxs = MOUTH_INDICES

        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)

    def process(self, frame: np.ndarray, thresholds: Mapping[str, float]) -> Tuple[np.ndarray, bool, Optional[float]]:
        """Process one frame and return the annotated frame, alarm flag, and risk score."""
        detection_thresholds = DetectionThresholds.from_mapping(thresholds)
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        drowsy_time_txt_pos = (10, int(frame_h // 2 * 1.7))
        alarm_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)
        risk_score: Optional[float] = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear, coordinates = evaluate_ocular_metrics(
                landmarks,
                self.eye_idxs["left"],
                self.eye_idxs["right"],
                frame_w,
                frame_h,
            )
            mar = compute_mouth_aspect_ratio(landmarks, self.mouth_idxs, frame_w, frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])

            mar_threshold = detection_thresholds.mar_threshold
            is_drowsy = ear < detection_thresholds.ear_threshold
            is_yawning = mar > mar_threshold

            if is_drowsy or is_yawning:
                end_time = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                self.state_tracker["start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if coordinates[0] is not None and coordinates[1] is not None:
                    x_coords = [coord[0] for coord in coordinates[0] + coordinates[1]]
                    y_coords = [coord[1] for coord in coordinates[0] + coordinates[1]]
                    cv2.rectangle(
                        frame,
                        (min(x_coords) - 20, min(y_coords) - 20),
                        (max(x_coords) + 20, max(y_coords) + 20),
                        self.RED,
                        4,
                    )

                if self.state_tracker["DROWSY_TIME"] >= detection_thresholds.wait_time:
                    self.state_tracker["play_alarm"] = True
                    if is_yawning:
                        event_type = "Yawn"
                        plot_text(frame, "YAWN ALERT!", alarm_txt_pos, self.state_tracker["COLOR"])
                    else:
                        event_type = "Micro-sleep"
                        plot_text(frame, "WAKE UP! WAKE UP", alarm_txt_pos, self.state_tracker["COLOR"])

                    log_incident(event_type, self.state_tracker["DROWSY_TIME"])
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            risk_score = compute_fatigue_risk_score(ear, mar)
            plot_text(frame, f"EAR: {round(ear, 2)}", self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, f"MAR: {round(mar, 2)}", (10, 55), self.state_tracker["COLOR"])
            plot_text(frame, f"Risk: {round(risk_score, 1)}%", (10, 80), self.state_tracker["COLOR"])
            plot_text(frame, f"Sustain: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs", drowsy_time_txt_pos, self.state_tracker["COLOR"])
        else:
            self.state_tracker["start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"], risk_score
