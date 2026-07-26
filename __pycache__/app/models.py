"""Typed data models for the HelpGuard AI surveillance application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import DEFAULT_THRESHOLDS


@dataclass(frozen=True)
class DetectionThresholds:
    """Configuration values that govern when the system raises an alert."""

    ear_threshold: float
    wait_time: float
    mar_threshold: float = DEFAULT_THRESHOLDS["MAR_THRESH"]

    @classmethod
    def from_mapping(cls, thresholds: Mapping[str, float]) -> "DetectionThresholds":
        """Create thresholds from a dictionary of slider values."""
        return cls(
            ear_threshold=float(thresholds.get("EAR_THRESH", DEFAULT_THRESHOLDS["EAR_THRESH"])),
            wait_time=float(thresholds.get("WAIT_TIME", DEFAULT_THRESHOLDS["WAIT_TIME"])),
            mar_threshold=float(thresholds.get("MAR_THRESH", DEFAULT_THRESHOLDS["MAR_THRESH"])),
        )

    def to_mapping(self) -> dict[str, float]:
        """Convert the thresholds back to a plain dictionary."""
        return {
            "EAR_THRESH": self.ear_threshold,
            "WAIT_TIME": self.wait_time,
            "MAR_THRESH": self.mar_threshold,
        }
