"""YOLOv8-based person detector for HelpGuard AI surveillance."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


class YOLOPersonDetector:
    """Run a YOLOv8 person detector alongside the existing fatigue detector."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model = None
        if YOLO is not None:
            self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int, Optional[list[dict[str, object]]]]:
        """Return the annotated frame, person count, and detection details."""
        if self.model is None:
            return frame, 0, None

        results = self.model(frame, stream=True, verbose=False)
        annotated_frame = frame.copy()
        detections: list[dict[str, object]] = []
        person_count = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"person {confidence:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence,
                })
                person_count += 1

        return annotated_frame, person_count, detections
