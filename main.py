"""
Entry point for the football analysis pipelines (see pipelines.py):

video
-> player detection
-> multi-object tracking
-> team classification
-> annotation (+ pitch projection with --interactive)
-> output video(s)

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

from app_ui import ApplicationUI
from config import DEFAULT_CLIP_PATH, DEFAULT_OUTPUT_PATH
from interactive_viewer import MultiViewPlayer
from pipelines import process_video, process_video_multiview
from video_io import download_youtube_clip


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
        default=DEFAULT_OUTPUT_PATH,
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
            output_path=DEFAULT_CLIP_PATH,
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
                        output_path=DEFAULT_CLIP_PATH,
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
