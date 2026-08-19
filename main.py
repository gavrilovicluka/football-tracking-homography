"""
End-to-end: (optionally download a YouTube clip) -> detect (YOLOv11)
-> track (ByteTrack) -> annotate -> save output video.

Usage:
    python main.py --youtube-url "https://youtube.com/watch?v=XXXXXXXX" --start 00:01:30 --duration 15
    python main.py --youtube-url "https://www.youtube.com/watch?v=93NnB1dBzwM" --start 00:00:35 --duration 15 

    python main.py --video-path downloads/clip.mp4

    Need to install:
        - winget install ffmpeg
        - irm https://deno.land/install.ps1 | iex
"""
import argparse
from pathlib import Path
import numpy as np

import supervision as sv
from tqdm import tqdm

from detector import FootballDetector, CLASS_NAMES
from tracker import PlayerTracker
from video_io import download_youtube_clip, read_frames, get_video_info, VideoWriter
from team_classifier import TeamClassifier, PLAYER_CLASS_ID, extract_crops

WEIGHTS_PATH = "models/football_players_yolo11s_best.pt"

# ball=gold, goalkeeper=blue, player=red, referee=purple - matches CLASS_NAMES order
CLASS_COLORS = sv.ColorPalette.from_hex(["#FFD700", "#00BFFF", "#FF4136", "#B10DC9"])

# team0=red, team1=blue, goalkeeper=yellow, referee=purple, ball=white
DISPLAY_COLORS = sv.ColorPalette.from_hex(["#FF4136", "#0074D9", "#FFDC00", "#B10DC9", "#FFFFFF"])

def get_display_color_index(class_id: int, team_id: int | None) -> int:
    """Maps a detection to a color-palette index: 0/1 for team, else class-based."""
    if class_id == PLAYER_CLASS_ID and team_id is not None:
        return team_id  # 0 or 1
    return {0: 4, 1: 2, 3: 3}[class_id]  # ball, goalkeeper, referee

def build_annotators():
    box_annotator = sv.BoxAnnotator(color=CLASS_COLORS, thickness=2)
    label_annotator = sv.LabelAnnotator(color=CLASS_COLORS, text_scale=0.5, text_thickness=1)
    return box_annotator, label_annotator

def make_labels(detections: sv.Detections, team_ids: np.ndarray | None) -> list[str]:
    labels = []
    for i, (class_id, tracker_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
        name = CLASS_NAMES[class_id]
        tid = f"#{tracker_id}" if tracker_id is not None else ""
        team_tag = f" T{team_ids[i]}" if (class_id == PLAYER_CLASS_ID and team_ids is not None) else ""
        labels.append(f"{name}{team_tag} {tid}")
    return labels

# def make_labels(detections: sv.Detections) -> list[str]:
#     labels = []
#     for class_id, tracker_id, conf in zip(
#         detections.class_id, detections.tracker_id, detections.confidence
#     ):
#         name = CLASS_NAMES[class_id]
#         tid = f"#{tracker_id}" if tracker_id is not None else ""
#         labels.append(f"{name} {tid} {conf:.2f}")
#     return labels

# We need to collect a sufficient set of player crops to use to train team classification model
def collect_fitting_crops(video_path: Path, detector: FootballDetector, n_samples: int = 30) -> list:
    """Runs detection (no tracking needed) on evenly-spaced sample frames across
    the clip, collecting player crops to fit the team classifier on."""
    info = get_video_info(video_path)
    sample_indices = set(np.linspace(0, info["frame_count"] - 1, n_samples, dtype=int))

    crops = []
    for i, frame in enumerate(read_frames(video_path)):
        if i not in sample_indices:
            continue
        detections = detector.detect(frame)
        player_mask = detections.class_id == PLAYER_CLASS_ID
        crops.extend(extract_crops(frame, detections.xyxy[player_mask]))

    return crops

def process_video(video_path: Path, output_path: Path, conf: float = 0.25, device: str = "cpu"):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(info["fps"])))
    
    print("Fitting team classifier on sample frames...")
    team_classifier = TeamClassifier(device=device)
    fitting_crops = collect_fitting_crops(video_path, detector, n_samples=30)
    team_classifier.fit(fitting_crops)
    print(f"Fitted on {len(fitting_crops)} player crops.")

    # box_annotator, label_annotator = build_annotators()
    box_annotator = sv.BoxAnnotator(color=DISPLAY_COLORS, thickness=2)
    label_annotator = sv.LabelAnnotator(color=DISPLAY_COLORS, text_scale=0.5, text_thickness=1)
    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])

    unique_ids = set()

    try:
        for frame in tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing"):
            detections = detector.detect(frame)
            detections = tracker.update(detections)

            if detections.tracker_id is not None:
                unique_ids.update(detections.tracker_id.tolist())

            # team classification, players only
            team_ids_full = np.full(len(detections), -1, dtype=int)
            player_mask = detections.class_id == PLAYER_CLASS_ID

            if player_mask.any():
                player_crops = extract_crops(frame, detections.xyxy[player_mask])
                raw_preds = team_classifier.predict(player_crops)
                stable_preds = team_classifier.assign_team_ids(
                    detections.tracker_id[player_mask], raw_preds
                )
                team_ids_full[player_mask] = stable_preds

            labels = make_labels(detections, team_ids_full)

            color_indices = np.array([
                get_display_color_index(cid, tid if tid != -1 else None)
                for cid, tid in zip(detections.class_id, team_ids_full)
            ])
            # sv.BoxAnnotator colors by detections.class_id by default - temporarily
            # substitute so it colors by team/display index instead
            # display_detections = detections.copy()
            # display_detections.class_id = color_indices
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

            writer.write(annotated_frame)
    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")
    print(
        "(Rough sanity check, not a formal metric: expect somewhere around "
        "22 players + ref(s) + ball if tracking stays stable. A much higher "
        "count usually means frequent ID switches from occlusions/re-entries.)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-url", type=str, default=None)
    parser.add_argument("--start", type=str, default=None, help="HH:MM:SS clip start (with --youtube-url)")
    parser.add_argument("--duration", type=int, default=15, help="Clip duration in seconds")
    parser.add_argument("--video-path", type=str, default=None, help="Use an already-downloaded video instead")
    parser.add_argument("--output", type=str, default="outputs/tracked_output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', '0' for GPU 0, etc.")
    args = parser.parse_args()

    if args.video_path:
        video_path = Path(args.video_path)
    elif args.youtube_url:
        video_path = download_youtube_clip(
            args.youtube_url,
            output_path="downloads/clip.mp4",
            start_time=args.start,
            duration=args.duration,
        )
    else:
        raise ValueError("Provide either --youtube-url or --video-path")

    process_video(video_path, Path(args.output), conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()
