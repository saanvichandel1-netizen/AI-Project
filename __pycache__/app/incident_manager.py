"""Incident persistence and screenshot capture for distress alerts."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class IncidentManager:
    """Persist alert events with metadata, screenshots, and explainable summaries."""

    def __init__(self, log_path: str = "incident_log.csv", screenshot_dir: str = "screenshots") -> None:
        self.log_path = Path(log_path)
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def save_incident(
        self,
        alert_type: str,
        confidence: float,
        risk_score: float,
        people_count: int,
        reason: str,
        detected_features: list[str],
        frame: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Store a new incident row and optionally save a screenshot."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        incident_id = f"{timestamp}-{alert_type.lower().replace(' ', '_')}"
        screenshot_path = None

        if frame is not None:
            screenshot_path = self.screenshot_dir / f"{incident_id}.png"
            cv2.imwrite(str(screenshot_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        row = {
            "timestamp": timestamp,
            "alert_type": alert_type,
            "confidence": round(float(confidence), 2),
            "risk_score": round(float(risk_score), 2),
            "people_count": int(people_count),
            "reason": reason,
            "detected_features": " | ".join(detected_features),
            "screenshot": str(screenshot_path) if screenshot_path is not None else "",
        }

        file_exists = self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return row

    def read_incidents(self) -> list[dict[str, Any]]:
        """Read all incident rows from disk."""
        if not self.log_path.exists():
            return []

        with self.log_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
