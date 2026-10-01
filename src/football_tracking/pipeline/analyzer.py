"""
Per-frame analysis shared by all pipelines:
detection -> tracking -> team classification -> (optional) pitch keypoints + homography projection.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import numpy as np
import supervision as sv

from football_tracking.config import (
    CLASSIFICATION_INTERVAL,
    PITCH_CONFIDENCE,
    PITCH_IMAGE_SIZE,
    PITCH_KEYPOINT_WEIGHTS_PATH,
    PLAYER_CROPS_SAMPLE_COUNT,
    WEIGHTS_PATH,
)
from football_tracking.schema import BALL_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID
from football_tracking.detection.players import FootballDetector
from football_tracking.detection.pitch import PitchLandmarkDetector
from football_tracking.geometry.projection import PitchProjector
from football_tracking.teams.classifier import TeamClassifier, collect_fitting_crops, extract_crops, valid_box_mask
from football_tracking.tracking.tracker import PlayerTracker


@dataclass
class FrameResult:
    detections: sv.Detections
    team_ids: np.ndarray        # per detection; -1 for non-players or unclassified players
    player_mask: np.ndarray
    landmarks_xy: np.ndarray    # last detected pitch landmarks; empty when pitch analysis is off
    pitch_xy: np.ndarray | None  # pitch coordinates aligned with projection_class_ids
    projection_class_ids: np.ndarray


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
        team_classifier: TeamClassifier | None = None,
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
        
        # The position history is used to smooth the projected positions of players and the ball over time, which can help reduce jitter in the visualization.
        self._position_history = defaultdict(lambda: deque(maxlen=6))

    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        # --------------------------
        # Detection and tracking
        # --------------------------
        with self.timer.measure("Detection"):
            detections = self.detector.detect(frame)

        # Extract the ball detections and pad their bounding boxes to ensure they are fully captured.
        ball_mask = detections.class_id == BALL_CLASS_ID
        if ball_mask.any():
            detections.xyxy[ball_mask] = sv.pad_boxes(
                xyxy=detections.xyxy[ball_mask], px=10
            )

        # Detect only players and referees, not the ball, for tracking. The ball is usually too fast and small to track reliably.
        non_ball_mask = detections.class_id != BALL_CLASS_ID
        non_ball_detections = detections[non_ball_mask]

        with self.timer.measure("Tracking"):
            non_ball_detections = self.tracker.update(non_ball_detections)

        # Merge back the tracked players/referees and the ball with padded bounding boxes
        ball_detections = detections[ball_mask]
        if len(ball_detections) > 0:
            ball_detections.tracker_id = np.full(len(ball_detections), -1, dtype=int)

        detections = sv.Detections.merge([non_ball_detections, ball_detections])

        # --------------------------
        # Team classification
        # --------------------------
        player_mask = detections.class_id == PLAYER_CLASS_ID

        team_ids = np.full(len(detections), -1, dtype=int)
        if self.team_classifier is not None:
            with self.timer.measure("Classification"):
                team_ids = self._classify_teams(frame, detections, player_mask)

        # --------------------------
        # Pitch projection
        # --------------------------
        projection_mask = np.isin(
            detections.class_id,
            [BALL_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID],
        )
        pitch_xy = None
        projection_class_ids = np.empty(0, dtype=int)
        if self.pitch_detector is not None and self.projector is not None:
            with self.timer.measure("Pitch projection"):
                pitch_xy = self._project_detections(
                    frame, frame_idx, detections, projection_mask
                )
                if pitch_xy is not None:
                    projection_class_ids = detections.class_id[projection_mask]

        return FrameResult(
            detections=detections,
            team_ids=team_ids,
            player_mask=player_mask,
            landmarks_xy=self._last_landmark_xy,
            pitch_xy=pitch_xy,
            projection_class_ids=projection_class_ids,
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

    def _project_detections(
        self,
        frame: np.ndarray,
        frame_idx: int,
        detections: sv.Detections,
        projection_mask: np.ndarray,
    ) -> np.ndarray | None:
        # Between detections the last homography and landmarks are reused
        if frame_idx % self.pitch_detect_interval == 0:
            landmark_xy, landmark_indices, _ = self.pitch_detector.detect(frame)
            self._last_landmark_xy = landmark_xy
            if len(landmark_xy) >= 4:
                self._have_homography = self.projector.update(landmark_xy, landmark_indices)

        if not self._have_homography or not projection_mask.any():
            return None

        projected_detections = detections[projection_mask]
        image_points = projected_detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )
        ball_mask = projected_detections.class_id == BALL_CLASS_ID
        if ball_mask.any():
            ball_boxes = projected_detections.xyxy[ball_mask]
            image_points[ball_mask] = (ball_boxes[:, :2] + ball_boxes[:, 2:]) / 2

        # return self.projector.transform_points(image_points)

        # Get raw 2D coordinates on the pitch
        raw_pitch_xy = self.projector.transform_points(image_points)

        # Time filtering (smoothing) of coordinates to prevent "jittering"
        tracker_ids = projected_detections.tracker_id
        smoothed_pitch_xy = raw_pitch_xy.copy()

        for i, tid in enumerate(tracker_ids):
            is_ball = ball_mask[i] if i < len(ball_mask) else False

            if is_ball:
                # For the ball, we use a shorter window to avoid lagging behind fast movement
                key = "ball"
                max_len = 3
            elif tid is not None and tid != -1:
                # For players with a known ID, we use a longer window to smooth their positions over time
                key = int(tid)
                max_len = 6
            else:
                # For objects without an ID, we skip smoothing
                continue

            # If the history for this key does not have a defined maxlen, we adjust it
            if self._position_history[key].maxlen != max_len:
                self._position_history[key] = deque(self._position_history[key], maxlen=max_len)

            # Add the new position to the history and compute the average
            self._position_history[key].append(raw_pitch_xy[i])
            smoothed_pitch_xy[i] = np.mean(np.array(self._position_history[key]), axis=0)

        return smoothed_pitch_xy


def create_analyzer(
    video_path: Path,
    fps: float,
    conf: float = 0.25,
    device: str = "cpu",
    with_pitch: bool = True ,
    with_classification: bool = False,
    pitch_conf: float = PITCH_CONFIDENCE,
    pitch_imgsz: int = PITCH_IMAGE_SIZE,
    pitch_detect_interval: int = 1,
) -> FrameAnalyzer:
    """Loads the models and fits the team classifier (if enabled) on sample frames of the video."""
    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(fps)))

    team_classifier = None
    if with_classification:
        print("Fitting team classifier on sample frames...")
        team_classifier = TeamClassifier(
            device=device, 
            n_teams=2, 
            classification_interval=CLASSIFICATION_INTERVAL
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
        team_classifier=team_classifier,
        pitch_detector=pitch_detector,
        projector=projector,
        pitch_detect_interval=pitch_detect_interval,
    )
