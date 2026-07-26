"""Image preprocessing utilities applied before AI inference."""

from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """Apply configurable image enhancement operations to a frame."""

    def __init__(self) -> None:
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def apply(
        self,
        frame: np.ndarray,
        enable_hist_equalization: bool = False,
        enable_clahe: bool = False,
        enable_gaussian_blur: bool = False,
        enable_median_blur: bool = False,
        enable_background_subtraction: bool = False,
        enable_morph_opening: bool = False,
        enable_morph_closing: bool = False,
    ) -> np.ndarray:
        """Return a processed version of the input frame."""
        processed = frame.copy()

        if processed.ndim == 3 and processed.shape[2] == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed

        if enable_hist_equalization:
            gray = cv2.equalizeHist(gray)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if enable_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray if gray.ndim == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if enable_gaussian_blur:
            processed = cv2.GaussianBlur(processed, (5, 5), 0)

        if enable_median_blur:
            processed = cv2.medianBlur(processed, 5)

        if enable_background_subtraction:
            fg_mask = self.background_subtractor.apply(frame)
            processed = cv2.bitwise_and(processed, processed, mask=fg_mask)

        if enable_morph_opening:
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        if enable_morph_closing:
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        return processed
