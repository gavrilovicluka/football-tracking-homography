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
from time import perf_counter
import traceback
import numpy as np

from tqdm import tqdm

from app_ui import ApplicationUI
from config import CLASSIFICATION_INTERVAL, PLAYER_CROPS_SAMPLE_COUNT, WEIGHTS_PATH
from constants import PLAYER_CLASS_ID
from detector import FootballDetector
from interactive_viewer import MultiViewPlayer
from multiview_pipeline import process_video_multiview
from rendering import annotate_frame, build_annotators
from tracker import PlayerTracker
from video_io import download_youtube_clip, read_frames, get_video_info, VideoWriter
from team_classifier import TeamClassifier, collect_fitting_crops, extract_crops


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

    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Process the full video and open the 4-panel review window"
    )

    return parser.parse_args()

def process_video(video_path: Path, output_path: Path, conf: float = 0.25, device: str = "cpu"):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(info["fps"])))
    
    print("Fitting team classifier on sample frames...")
    team_classifier = TeamClassifier(
        device=device,
        n_teams=2,
        classification_interval=CLASSIFICATION_INTERVAL
    )
    fitting_crops = collect_fitting_crops(video_path, detector, n_samples=PLAYER_CROPS_SAMPLE_COUNT)
    team_classifier.fit(fitting_crops)
    print(f"Fitted on {len(fitting_crops)} player crops.")

    annotators = build_annotators()

    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])

    unique_ids = set()

    try:
        detection_time = 0.0
        tracking_time = 0.0
        classification_time = 0.0
        annotation_time = 0.0
        for frame in tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing"):
            start = perf_counter()

            detections = detector.detect(frame)

            detection_time += perf_counter() - start

            start = perf_counter()
    
            detections = tracker.update(detections)

            tracking_time += perf_counter() - start

            if detections.tracker_id is not None:
                unique_ids.update(detections.tracker_id.tolist())

            # team classification, players only
            team_ids_full = np.full(len(detections), -1, dtype=int)
            player_mask = detections.class_id == PLAYER_CLASS_ID

            start = perf_counter()

            if player_mask.any():
                player_crops = extract_crops(frame, detections.xyxy[player_mask])
                stable_preds = team_classifier.predict_tracked(
                    player_crops,
                    detections.tracker_id[player_mask]
                )
                team_ids_full[player_mask] = stable_preds

            classification_time += (
                perf_counter() - start
            )

            start = perf_counter()

            annotated_frame = annotate_frame(frame, detections, team_ids_full, annotators)

            writer.write(annotated_frame)

            annotation_time += (
                perf_counter() - start
            )
    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")
    print(
        "(Rough sanity check, not a formal metric: expect somewhere around "
        "22 players + ref(s) + ball if tracking stays stable. A much higher "
        "count usually means frequent ID switches from occlusions/re-entries.)"
    )

    print("\nTiming summary:")
    print(
        f"Detection:       {detection_time:.2f}s"
    )
    print(
        f"Tracking:        {tracking_time:.2f}s"
    )
    print(
        f"Classification:  {classification_time:.2f}s"
    )
    print(
        f"Annotation/write:{annotation_time:.2f}s"
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

    if args.interactive:
        paths = process_video_multiview(
            video_path=video_path,
            output_dir=args.output.parent,
            conf=args.conf,
            device=args.device,
        )
        MultiViewPlayer(paths).run()
    else:
        process_video(video_path=video_path, output_path=args.output, conf=args.conf, device=args.device)


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


if __name__ == "__main__":
    main()
