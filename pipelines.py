"""
Video pipelines built on FrameAnalyzer:

process_video            - single annotated output video (detections, tracks, teams)
process_video_multiview  - four synchronized videos plus a combined 2x2 grid:
    original    - raw input frame
    annotated   - player/ball detection + tracking + team-colored boxes
    keypoints   - pitch keypoint detections drawn on the frame
    projection  - 2D top-down pitch with players projected via homography

The multiview outputs are aligned 1:1 by frame index, so
interactive_viewer.MultiViewPlayer can scrub them together.
"""
from pathlib import Path

import cv2
from tqdm import tqdm

from analyzer import create_analyzer
from config import (
    LABEL_HEIGHT,
    PITCH_CONFIDENCE,
    PITCH_DETECT_INTERVAL,
    PITCH_IMAGE_SIZE,
    multiview_output_paths,
)
from rendering import (
    annotate_frame,
    build_annotators,
    compose_grid,
    draw_keypoints,
    draw_projection,
)
from video_io import read_frames, get_video_info, VideoWriter


def process_video(video_path: Path, output_path: Path, conf: float = 0.25, device: str = "cpu"):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    analyzer = create_analyzer(video_path, fps=info["fps"], conf=conf, device=device)
    annotators = build_annotators()
    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])

    unique_ids = set()

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing")
        ):
            result = analyzer.process(frame, frame_idx)

            if result.detections.tracker_id is not None:
                unique_ids.update(result.detections.tracker_id.tolist())

            with analyzer.timer.measure("Annotation/write"):
                writer.write(annotate_frame(frame, result.detections, result.team_ids, annotators))
    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")
    print(
        "(Rough sanity check, not a formal metric: expect somewhere around "
        "22 players + ref(s) + ball if tracking stays stable. A much higher "
        "count usually means frequent ID switches from occlusions/re-entries.)"
    )
    analyzer.timer.report()


def process_video_multiview(
    video_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    device: str = "cpu",
    pitch_conf: float = PITCH_CONFIDENCE,
    pitch_imgsz: int = PITCH_IMAGE_SIZE,
    pitch_detect_interval: int = PITCH_DETECT_INTERVAL,
) -> dict[str, Path]:
    """
    Runs the full analysis over the whole video once and writes the
    multiview outputs into `output_dir`. Returns their paths keyed
    "original", "annotated", "keypoints", "projection", "combined".

    pitch_detect_interval: run the (usually expensive) pitch-keypoint
    detector every N frames instead of every frame, reusing the last
    known homography and keypoint overlay in between. Set to 1 for
    per-frame accuracy; raise it (e.g. 5) if this stage dominates runtime.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {info['frame_count']} frames")

    analyzer = create_analyzer(
        video_path,
        fps=fps,
        conf=conf,
        device=device,
        with_pitch=True,
        pitch_conf=pitch_conf,
        pitch_imgsz=pitch_imgsz,
        pitch_detect_interval=pitch_detect_interval,
    )
    pitch_config = analyzer.projector.config
    annotators = build_annotators()

    paths = multiview_output_paths(output_dir)
    writers = {
        name: VideoWriter(paths[name], fps=fps, width=width, height=height)
        for name in ("original", "annotated", "keypoints", "projection")
    }

    tile_w, tile_h = width // 2, height // 2
    combined_w, combined_h = tile_w * 2, (tile_h + LABEL_HEIGHT) * 2
    writers["combined"] = VideoWriter(paths["combined"], fps=fps, width=combined_w, height=combined_h)

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing")
        ):
            result = analyzer.process(frame, frame_idx)

            with analyzer.timer.measure("Annotation/write"):
                annotated_frame = annotate_frame(frame, result.detections, result.team_ids, annotators)
                keypoints_frame = draw_keypoints(frame, result.landmarks_xy)
                pitch = draw_projection(pitch_config, result.pitch_xy, result.team_ids[result.player_mask])
                projection_frame = cv2.resize(pitch, (width, height))

                writers["original"].write(frame)
                writers["annotated"].write(annotated_frame)
                writers["keypoints"].write(keypoints_frame)
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
    analyzer.timer.report()

    return paths
