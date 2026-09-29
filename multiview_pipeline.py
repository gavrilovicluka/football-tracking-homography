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
from constants import PLAYER_CLASS_ID
from detector import FootballDetector
from rendering import (
    LABEL_HEIGHT,
    annotate_frame,
    build_annotators,
    compose_grid,
    draw_keypoints,
    draw_projection,
)
from tracker import PlayerTracker
from team_classifier import TeamClassifier, collect_fitting_crops, extract_crops
from pitch_landmark_detector import PitchLandmarkDetector
from pitch_projection import PitchProjector
from video_io import read_frames, get_video_info, VideoWriter


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

    annotators = build_annotators()

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

            annotated_frame = annotate_frame(frame, detections, team_ids_full, annotators)
            writers["annotated"].write(annotated_frame)

            # ---- pitch keypoints (optionally throttled) ----
            if frame_idx % pitch_detect_interval == 0:
                landmark_xy, landmark_indices, confidences = pitch_detector.detect(frame)
                last_landmark_xy = landmark_xy
                if len(landmark_xy) >= 4:
                    have_homography = projector.update(landmark_xy, landmark_indices)

            keypoints_frame = draw_keypoints(frame, last_landmark_xy)
            writers["keypoints"].write(keypoints_frame)

            # ---- 2D projection ----
            pitch_points = None
            if have_homography and player_mask.any():
                player_points = detections[player_mask].get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                pitch_points = projector.transform_points(player_points)
            pitch = draw_projection(projector.config, pitch_points, team_ids_full[player_mask])
            projection_frame = cv2.resize(pitch, (width, height))
            writers["projection"].write(projection_frame)

            writers["combined"].write(compose_grid(
                {
                    "original": frame,
                    "annotated": annotated_frame,
                    "keypoints": keypoints_frame,
                    "projection": projection_frame,
                },
                tile_w,
                tile_h,
            ))
    finally:
        for writer in writers.values():
            writer.release()

    print("Saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return paths
