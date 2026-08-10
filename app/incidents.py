"""Helpers for storing and exporting incident logs for HelpGuard AI."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "incidents.csv"
CSV_COLUMNS = ["timestamp", "ear", "mar", "risk_score", "status"]


def ensure_log_file() -> Path:
    """Create the incident log directory and CSV file if they do not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
    return LOG_FILE


def append_incident(entry: Dict[str, object]) -> Path:
    """Append a single incident record to the CSV log file."""
    path = ensure_log_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(
            {
                "timestamp": entry.get("timestamp", ""),
                "ear": entry.get("ear", ""),
                "mar": entry.get("mar", ""),
                "risk_score": entry.get("risk_score", ""),
                "status": entry.get("status", ""),
            }
        )
    return path


def build_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
    """Create a CSV payload for download from a list of incident rows."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "timestamp": row.get("timestamp", ""),
                "ear": row.get("ear", ""),
                "mar": row.get("mar", ""),
                "risk_score": row.get("risk_score", ""),
                "status": row.get("status", ""),
            }
        )
    return buffer.getvalue().encode("utf-8")
