"""Configuration values for the driver safety monitoring application."""

from __future__ import annotations

from pathlib import Path

LOG_PATH = "incident_log.csv"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_VIDEO_PATH = PROJECT_ROOT / "test_video.mp4"

DEFAULT_THRESHOLDS = {
    "EAR_THRESH": 0.18,
    "WAIT_TIME": 1.0,
    "MAR_THRESH": 0.5,
}

EYE_INDICES = {
    "left": [362, 385, 387, 263, 373, 380],
    "right": [33, 160, 158, 133, 153, 144],
}

MOUTH_INDICES = [78, 308, 13, 14]
