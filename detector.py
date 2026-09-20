"""
Wraps the fine-tuned YOLOv11 model for football object detection
(ball, goalkeeper, player, referee). Returns supervision.Detections,
which is the format tracker.py and main.py expect.
"""
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

from constants import DEFAULT_CLASS_CONF_THRESHOLDS

class FootballDetector:
    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.10,
        imgsz: int = 1280,
        device: str = "cpu",
        class_conf_thresholds: dict | None = None,
    ):
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.class_conf_thresholds = class_conf_thresholds or DEFAULT_CLASS_CONF_THRESHOLDS

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        # return sv.Detections.from_ultralytics(result)
        detections = sv.Detections.from_ultralytics(result)
        return self._filter_by_class_conf(detections)

    def _filter_by_class_conf(self, detections: sv.Detections) -> sv.Detections:
        if len(detections) == 0:
            return detections
        keep = np.array([
            conf >= self.class_conf_thresholds.get(int(cls_id), self.conf)
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ])
        return detections[keep]
