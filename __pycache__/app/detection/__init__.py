"""Detection package exports for the HelpGuard AI application."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .distress_engine import DistressEngine
from .fall_detector import MediaPipeFallDetector
from .gesture_detector import MediaPipeGestureDetector
from .pose_detector import MediaPipePoseDetector

legacy_module_path = Path(__file__).resolve().parent.parent / "detection.py"
legacy_spec = importlib.util.spec_from_file_location("app._legacy_detection", legacy_module_path)
if legacy_spec is None or legacy_spec.loader is None:
    raise ImportError(f"Unable to load legacy detection module from {legacy_module_path}")

legacy_module = importlib.util.module_from_spec(legacy_spec)
legacy_spec.loader.exec_module(legacy_module)

SafetyStateProcessor = legacy_module.SafetyStateProcessor

__all__ = ["SafetyStateProcessor", "MediaPipePoseDetector", "MediaPipeFallDetector", "MediaPipeGestureDetector", "DistressEngine"]
