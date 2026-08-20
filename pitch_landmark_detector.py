from pathlib import Path

import numpy as np
from ultralytics import YOLO


class PitchLandmarkDetector:
    def __init__(
        self,
        weights_path: str | Path,
        conf: float = 0.5,
        imgsz: int = 960,
        device: str = "cpu",
    ):
        self.model = YOLO(str(weights_path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        result = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=np.float32),
            )

        xyxy = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Keep only highest-confidence detection for each landmark class
        best_by_class = {}

        for box, class_id, confidence in zip(
            xyxy,
            class_ids,
            confidences,
        ):
            current = best_by_class.get(class_id)

            if current is None or confidence > current[1]:
                best_by_class[class_id] = (box, confidence)

        points = []
        indices = []
        confs = []

        for class_id in sorted(best_by_class):
            box, confidence = best_by_class[class_id]

            x1, y1, x2, y2 = box

            points.append([
                (x1 + x2) / 2,
                (y1 + y2) / 2,
            ])

            indices.append(class_id)
            confs.append(confidence)

        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(indices, dtype=int),
            np.asarray(confs, dtype=np.float32),
        )