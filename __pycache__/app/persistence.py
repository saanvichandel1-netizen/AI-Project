"""Persistence helpers for surveillance incident logging."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

from .config import LOG_PATH


def append_incident(event_type: str, duration_seconds: float, log_path: str = LOG_PATH) -> None:
    """Persist an alert event to a local CSV file for later analysis."""
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "event_type", "duration_seconds"])
        writer.writerow([datetime.utcnow().isoformat(timespec="seconds"), event_type, round(duration_seconds, 3)])


def read_incident_log_bytes(log_path: str = LOG_PATH) -> Optional[bytes]:
    """Read the incident log as bytes when present, otherwise return None."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "rb") as log_file:
        return log_file.read()
