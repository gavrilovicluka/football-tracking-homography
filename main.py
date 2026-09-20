"""
End-to-end football player analysis pipeline:

video
-> player detection
-> multi-object tracking
-> team classification
-> annotation
-> output video

Usage:
    python main.py

    python main.py --youtube-url "https://youtube.com/watch?v=XXXXXXXX" --start 00:01:30 --duration 15
    python main.py --youtube-url "https://www.youtube.com/watch?v=93NnB1dBzwM" --start 00:00:35 --duration 15 

    python main.py --video-path downloads/clip.mp4

    Need to install:
        - winget install ffmpeg
        - irm https://deno.land/install.ps1 | iex
"""
import argparse
from pathlib import Path
import traceback
import numpy as np

import supervision as sv
from tqdm import tqdm

from app_ui import ApplicationUI
from config import PLAYER_CROPS_SAMPLE_COUNT, WEIGHTS_PATH
from constants import CLASS_NAMES, DISPLAY_COLOR_INDEX, DISPLAY_COLORS, PLAYER_CLASS_ID
from detector import FootballDetector
from tracker import PlayerTracker
from utils import collect_fitting_crops
from video_io import download_youtube_clip, read_frames, get_video_info, VideoWriter
from team_classifier import TeamClassifier, extract_crops


def parse_arguments():
    parser = argparse.ArgumentParser()

    source_group = parser.add_mutually_exclusive_group()

    source_group.add_argument(
        "--youtube-url",
        type=str,
    )

    source_group.add_argument(
        "--video-path",
        type=Path,
        help="Use an already downloaded video.",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="HH:MM:SS clip start (with --youtube-url)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Clip duration in seconds",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tracked_output.mp4"),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu', '0' for GPU 0, etc.",
    )

    return parser.parse_args()

def get_display_color_index(class_id: int, team_id: int | None) -> int:
    """
    Maps a detection to a color-palette index: 0/1 for team, else class-based.
    """
    if class_id == PLAYER_CLASS_ID and team_id is not None:
        return team_id  # 0 or 1
    
    return DISPLAY_COLOR_INDEX[class_id]  # ball, goalkeeper, referee

def build_annotators():
    box_annotator = sv.BoxAnnotator(
        color=DISPLAY_COLORS, 
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        color=DISPLAY_COLORS,
        text_scale=0.5,
        text_thickness=1,
    )

    return box_annotator, label_annotator

def make_labels(detections: sv.Detections, team_ids: np.ndarray | None) -> list[str]:
    labels = []
    for i, (class_id, tracker_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
        name = CLASS_NAMES[class_id]
        tid = f"#{tracker_id}" if tracker_id is not None else ""
        team_tag = f" T{team_ids[i]}" if (class_id == PLAYER_CLASS_ID and team_ids is not None) else ""
        labels.append(f"{name}{team_tag} {tid}")
    return labels

def process_video(video_path: Path, output_path: Path, conf: float = 0.25, device: str = "cpu"):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(info["fps"])))
    
    print("Fitting team classifier on sample frames...")
    team_classifier = TeamClassifier(device=device)
    fitting_crops = collect_fitting_crops(video_path, detector, n_samples=PLAYER_CROPS_SAMPLE_COUNT)
    team_classifier.fit(fitting_crops)
    print(f"Fitted on {len(fitting_crops)} player crops.")

    box_annotator, label_annotator = build_annotators()

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
    args = parse_arguments()

    if args.video_path or args.youtube_url:
        run_from_cli(args)
    else:
        run_from_ui()

def run_from_cli(args):
    if args.video_path:
        video_path = args.video_path
    else:
        video_path = download_youtube_clip(
            args.youtube_url,
            output_path="downloads/clip.mp4",
            start_time=args.start,
            duration=args.duration,
        )

    process_video(
        video_path=video_path,
        output_path=args.output,
        conf=args.conf,
        device=args.device,
    )


def run_from_ui():
    app = ApplicationUI()
    try:
        while True:
            args = app.run()

            if args is None:
                break

            try:
                if args["source_type"] == "file":
                    video_path = args["video_path"]
                else:
                    video_path = download_youtube_clip(
                        args["youtube_url"],
                        output_path="downloads/clip.mp4",
                        start_time=args["start"],
                        duration=args["duration"],
                    )

                process_video(
                    video_path=video_path,
                    output_path=args["output"],
                    conf=args["conf"],
                    device=args["device"],
                )

                app.show_info(
                    "Processing complete",
                    f"Video saved to:\n{args['output']}",
                )

                break

            except Exception as error:
                traceback.print_exc()

                app.show_error(
                    "Processing failed",
                    str(error),
                )
    finally:
        app.close()


    # process_video(video_path, args.output, conf=args.conf, device=args.device)


if __name__ == "__main__":
    main()
