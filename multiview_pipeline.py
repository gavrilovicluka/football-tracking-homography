"""
Combined per-frame pipeline that produces four synchronized video streams:

1. original    - raw input frame
2. annotated   - player/ball detection + tracking + team-colored boxes
3. keypoints   - pitch keypoint detections drawn on the frame
4. projection  - 2D top-down pitch with players projected via homography

All four are written frame-by-frame to four .mp4 files in `output_dir`,
already aligned 1:1 by frame index, so interactive_viewer.MultiViewPlayer
can scrub them together.

Assumes the same module interfaces already used in main.py and
visualize_player_projection.py (FootballDetector, PlayerTracker,
TeamClassifier, PitchLandmarkDetector, PitchProjector, VideoWriter, etc).
Adjust the imports below if any of those signatures differ in your repo.
"""
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from config import (
    CLASSIFICATION_INTERVAL,
    PITCH_KEYPOINT_WEIGHTS_PATH,
    PLAYER_CROPS_SAMPLE_COUNT,
    WEIGHTS_PATH,
)
from constants import (
    CLASS_NAMES,
    DISPLAY_COLOR_INDEX,
    DISPLAY_COLORS,
    LABEL_HEIGHT,
    PLAYER_CLASS_ID,
    TEAM_COLORS,
    TILE_TITLES,
)
from detector import FootballDetector
from tracker import PlayerTracker
from team_classifier import TeamClassifier, extract_crops
from pitch_landmark_detector import PitchLandmarkDetector
from pitch_projection import PitchProjector
from utils import collect_fitting_crops
from video_io import read_frames, get_video_info, VideoWriter

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch


def get_display_color_index(class_id: int, team_id: int | None) -> int:
    if class_id == PLAYER_CLASS_ID and team_id is not None:
        return team_id
    return DISPLAY_COLOR_INDEX[class_id]


def build_annotators():
    box_annotator = sv.BoxAnnotator(color=DISPLAY_COLORS, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=DISPLAY_COLORS, text_scale=0.5, text_thickness=1
    )
    return box_annotator, label_annotator


def make_labels(detections, team_ids_full):
    labels = []
    for class_id, tracker_id, team_id in zip(
        detections.class_id, detections.tracker_id, team_ids_full
    ):
        name = CLASS_NAMES[class_id]
        tid = f"#{tracker_id}" if tracker_id is not None else ""
        team_tag = f" T{team_id}" if (class_id == PLAYER_CLASS_ID and team_id != -1) else ""
        labels.append(f"{name}{team_tag} {tid}")
    return labels

def label_tile(frame, text):
    bar = np.zeros((LABEL_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([bar, frame])

def process_video_multiview(
    video_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    device: str = "cpu",
    pitch_conf: float = 0.5,
    pitch_imgsz: int = 960,
    pitch_detect_interval: int = 1,
) -> dict[str, Path]:
    """
    Runs detection + tracking + team classification + pitch keypoint
    detection + homography projection over the whole video, once.

    pitch_detect_interval: run the (usually expensive) pitch-keypoint
    detector every N frames instead of every frame, reusing the last
    known homography and keypoint overlay in between. Set to 1 for
    per-frame accuracy; raise it (e.g. 5) if this stage dominates runtime.

    Writes four synchronized .mp4 files into `output_dir` and returns
    their paths, keyed "original", "annotated", "keypoints", "projection".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {info['frame_count']} frames")

    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(fps)))

    print("Fitting team classifier on sample frames...")
    team_classifier = TeamClassifier(
        device=device, n_teams=2, classification_interval=CLASSIFICATION_INTERVAL
    )
    fitting_crops = collect_fitting_crops(
        video_path, detector, n_samples=PLAYER_CROPS_SAMPLE_COUNT
    )
    team_classifier.fit(fitting_crops)
    print(f"Fitted on {len(fitting_crops)} player crops.")

    pitch_detector = PitchLandmarkDetector(
        weights_path=PITCH_KEYPOINT_WEIGHTS_PATH,
        conf=pitch_conf,
        imgsz=pitch_imgsz,
        device=device,
    )
    projector = PitchProjector()

    box_annotator, label_annotator = build_annotators()

    paths = {
        "original": output_dir / "original.mp4",
        "annotated": output_dir / "annotated.mp4",
        "keypoints": output_dir / "keypoints.mp4",
        "projection": output_dir / "projection.mp4",
    }
    writers = {
        name: VideoWriter(path, fps=fps, width=width, height=height)
        for name, path in paths.items()
    }

    tile_w, tile_h = width // 2, height // 2
    combined_w, combined_h = tile_w * 2, (tile_h + LABEL_HEIGHT) * 2

    paths["combined"] = output_dir / "combined.mp4"
    writers["combined"] = VideoWriter(paths["combined"], fps=fps, width=combined_w, height=combined_h)

    have_homography = False
    last_landmark_xy = []

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing")
        ):
            writers["original"].write(frame)

            # ---- detection + tracking + team classification ----
            detections = detector.detect(frame)
            detections = tracker.update(detections)

            team_ids_full = np.full(len(detections), -1, dtype=int)
            player_mask = detections.class_id == PLAYER_CLASS_ID
            if player_mask.any():
                player_crops = extract_crops(frame, detections.xyxy[player_mask])
                team_ids_full[player_mask] = team_classifier.predict_tracked(
                    player_crops, detections.tracker_id[player_mask]
                )

            labels = make_labels(detections, team_ids_full)
            color_indices = np.array([
                get_display_color_index(cid, tid if tid != -1 else None)
                for cid, tid in zip(detections.class_id, team_ids_full)
            ])
            display_detections = sv.Detections(
                xyxy=detections.xyxy.copy(),
                mask=detections.mask.copy() if detections.mask is not None else None,
                confidence=detections.confidence.copy() if detections.confidence is not None else None,
                class_id=color_indices,
                tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
            )
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=display_detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=display_detections, labels=labels)
            writers["annotated"].write(annotated_frame)

            # ---- pitch keypoints (optionally throttled) ----
            if frame_idx % pitch_detect_interval == 0:
                landmark_xy, landmark_indices, confidences = pitch_detector.detect(frame)
                last_landmark_xy = landmark_xy
                if len(landmark_xy) >= 4:
                    have_homography = projector.update(landmark_xy, landmark_indices)

            keypoints_frame = frame.copy()
            for (x, y) in last_landmark_xy:
                cv2.circle(keypoints_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            writers["keypoints"].write(keypoints_frame)

            # ---- 2D projection ----
            pitch = draw_pitch(projector.config)
            if have_homography and player_mask.any():
                player_points = detections[player_mask].get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                pitch_points = projector.transform_points(player_points)
                player_teams = team_ids_full[player_mask]
                for team_id in [0, 1]:
                    team_mask = player_teams == team_id
                    if team_mask.any():
                        pitch = draw_points_on_pitch(
                            config=projector.config,
                            xy=pitch_points[team_mask],
                            face_color=TEAM_COLORS[team_id],
                            edge_color=sv.Color.BLACK,
                            radius=10,
                            pitch=pitch,
                        )
            projection_frame = cv2.resize(pitch, (width, height))
            writers["projection"].write(projection_frame)

            # ---- combined view ----
            tiles = [
                label_tile(cv2.resize(frame, (tile_w, tile_h)), TILE_TITLES["original"]),
                label_tile(cv2.resize(annotated_frame, (tile_w, tile_h)), TILE_TITLES["annotated"]),
                label_tile(cv2.resize(keypoints_frame, (tile_w, tile_h)), TILE_TITLES["keypoints"]),
                label_tile(cv2.resize(projection_frame, (tile_w, tile_h)), TILE_TITLES["projection"]),
            ]
            top = cv2.hconcat([tiles[0], tiles[1]])
            bottom = cv2.hconcat([tiles[2], tiles[3]])
            writers["combined"].write(cv2.vconcat([top, bottom]))
    finally:
        for writer in writers.values():
            writer.release()

    print("Saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return paths
