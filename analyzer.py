"""
Per-frame analysis shared by all pipelines:
detection -> tracking -> team classification -> (optional) pitch keypoints + homography projection.
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np
import supervision as sv

from config import (
    CLASSIFICATION_INTERVAL,
    PITCH_CONFIDENCE,
    PITCH_IMAGE_SIZE,
    PITCH_KEYPOINT_WEIGHTS_PATH,
    PLAYER_CROPS_SAMPLE_COUNT,
    WEIGHTS_PATH,
)
from constants import PLAYER_CLASS_ID
from detector import FootballDetector
from pitch_landmark_detector import PitchLandmarkDetector
from pitch_projection import PitchProjector
from team_classifier import TeamClassifier, collect_fitting_crops, extract_crops, valid_box_mask
from tracker import PlayerTracker


@dataclass
class FrameResult:
    detections: sv.Detections
    team_ids: np.ndarray        # per detection; -1 for non-players or unclassified players
    player_mask: np.ndarray
    landmarks_xy: np.ndarray    # last detected pitch landmarks; empty when pitch analysis is off
    pitch_xy: np.ndarray | None  # pitch coordinates of detections[player_mask]; None without homography


class StageTimer:
    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.totals[stage] += perf_counter() - start

    def report(self) -> None:
        print("\nTiming summary:")
        for stage, total in self.totals.items():
            print(f"{stage + ':':<18}{total:.2f}s")


class FrameAnalyzer:
    def __init__(
        self,
        detector: FootballDetector,
        tracker: PlayerTracker,
        team_classifier: TeamClassifier,
        pitch_detector: PitchLandmarkDetector | None = None,
        projector: PitchProjector | None = None,
        pitch_detect_interval: int = 1,
    ):
        self.detector = detector
        self.tracker = tracker
        self.team_classifier = team_classifier
        self.pitch_detector = pitch_detector
        self.projector = projector
        self.pitch_detect_interval = pitch_detect_interval
        self.timer = StageTimer()

        self._have_homography = False
        self._last_landmark_xy = np.empty((0, 2), dtype=np.float32)

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        with self.timer.measure("Detection"):
            detections = self.detector.detect(frame)

        with self.timer.measure("Tracking"):
            detections = self.tracker.update(detections)

        player_mask = detections.class_id == PLAYER_CLASS_ID

        with self.timer.measure("Classification"):
            team_ids = self._classify_teams(frame, detections, player_mask)

        pitch_xy = None
        if self.pitch_detector is not None and self.projector is not None:
            with self.timer.measure("Pitch projection"):
                pitch_xy = self._project_players(frame, frame_idx, detections, player_mask)

        return FrameResult(
            detections=detections,
            team_ids=team_ids,
            player_mask=player_mask,
            landmarks_xy=self._last_landmark_xy,
            pitch_xy=pitch_xy,
        )

    def _classify_teams(
        self, frame: np.ndarray, detections: sv.Detections, player_mask: np.ndarray
    ) -> np.ndarray:
        team_ids = np.full(len(detections), -1, dtype=int)

        # Degenerate boxes produce no crop, so exclude them to keep crops aligned with detections
        classify_mask = player_mask.copy()
        classify_mask[player_mask] = valid_box_mask(frame.shape, detections.xyxy[player_mask])

        if classify_mask.any():
            crops = extract_crops(frame, detections.xyxy[classify_mask])
            team_ids[classify_mask] = self.team_classifier.predict_tracked(
                crops, detections.tracker_id[classify_mask]
            )
        return team_ids

    def _project_players(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: sv.Detections,
        player_mask: np.ndarray,
    ) -> np.ndarray | None:
        # Between detections the last homography and landmarks are reused
        if frame_idx % self.pitch_detect_interval == 0:
            landmark_xy, landmark_indices, _ = self.pitch_detector.detect(frame)
            self._last_landmark_xy = landmark_xy
            if len(landmark_xy) >= 4:
                self._have_homography = self.projector.update(landmark_xy, landmark_indices)

        if not self._have_homography or not player_mask.any():
            return None

        player_points = detections[player_mask].get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
        return self.projector.transform_points(player_points)


def create_analyzer(
    video_path: Path,
    fps: float,
    conf: float = 0.25,
    device: str = "cpu",
    with_pitch: bool = False,
    pitch_conf: float = PITCH_CONFIDENCE,
    pitch_imgsz: int = PITCH_IMAGE_SIZE,
    pitch_detect_interval: int = 1,
) -> FrameAnalyzer:
    """Loads the models and fits the team classifier on sample frames of the video."""
    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(fps)))

    print("Fitting team classifier on sample frames...")
    team_classifier = TeamClassifier(
        device=device, n_teams=2, classification_interval=CLASSIFICATION_INTERVAL
    )
    fitting_crops = collect_fitting_crops(video_path, detector, n_samples=PLAYER_CROPS_SAMPLE_COUNT)
    team_classifier.fit(fitting_crops)
    print(f"Fitted on {len(fitting_crops)} player crops.")

    pitch_detector = None
    projector = None
    if with_pitch:
        pitch_detector = PitchLandmarkDetector(
            weights_path=PITCH_KEYPOINT_WEIGHTS_PATH,
            conf=pitch_conf,
            imgsz=pitch_imgsz,
            device=device,
        )
        projector = PitchProjector()

    return FrameAnalyzer(
        detector,
        tracker,
        team_classifier,
        pitch_detector=pitch_detector,
        projector=projector,
        pitch_detect_interval=pitch_detect_interval,
    )
