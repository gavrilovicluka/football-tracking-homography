"""
Wraps a YOLO-pose model trained to detect pitch landmark keypoints (a single
"pitch" instance per frame, with N keypoints - corners, box lines, spots, etc.).
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PitchKeypointDetector:
    def __init__(
            self, 
            weights_path: str | Path, 
            conf: float = 0.5, 
            imgsz: int = 960, 
            device: str = "cpu"
    ):
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (keypoints_xy, keypoints_conf):
          keypoints_xy:   (K, 2) pixel coordinates, one row per pitch landmark
          keypoints_conf: (K,)   confidence per landmark

        If no pitch instance is detected in the frame, returns empty arrays.
        """
        result = self.model.predict(
            source=frame, 
            conf=self.conf, 
            imgsz=self.imgsz, 
            device=self.device, 
            verbose=False,
        )[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return np.empty((0, 2)), np.empty((0,))

        # take the first (highest-confidence) detected pitch instance
        keypoints_xy = result.keypoints.xy[0].cpu().numpy()
        keypoints_conf = (
            result.keypoints.conf[0].cpu().numpy()
            if result.keypoints.conf is not None
            else np.ones(len(keypoints_xy))
        )
        return keypoints_xy, keypoints_conf

    def detect_filtered(self, frame: np.ndarray, min_conf: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Same as detect(), but returns only keypoints above min_conf, plus their
        original indices (needed to look up matching real-world coordinates)."""
        keypoints_xy, keypoints_conf = self.detect(frame)
        if len(keypoints_xy) == 0:
            return np.empty((0, 2)), np.empty((0,), dtype=int)
        mask = keypoints_conf >= min_conf
        indices = np.where(mask)[0]
        return keypoints_xy[mask], indices