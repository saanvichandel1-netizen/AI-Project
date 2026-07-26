"""Video-loading helpers for the HelpGuard AI surveillance application."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2

from .config import LOCAL_VIDEO_PATH

PathLike = Union[str, Path]


def resolve_video_path(video_path: Optional[PathLike] = None) -> Path:
    """Return the local video path used for playback."""
    return Path(video_path) if video_path is not None else LOCAL_VIDEO_PATH


def open_video_capture(video_path: PathLike) -> Optional[cv2.VideoCapture]:
    """Open the local video capture and return None if the file cannot be read."""
    capture = cv2.VideoCapture(str(resolve_video_path(video_path)))
    return capture if capture.isOpened() else None
