"""
Video I/O helpers: download a YouTube clip with yt-dlp, and read/write frames.
"""
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def download_youtube_clip(
    url: str,
    output_path: str | Path,
    start_time: str | None = None,
    duration: int | None = None,
) -> Path:
    """
    Downloads only the requested segment of a YouTube video via yt-dlp's
    download_ranges (ffmpeg pulls just that time window, not the whole video -
    important for full-match footage that can run 1-2+ hours).
    Requires: pip install -U yt-dlp, and ffmpeg + a JS runtime (deno) on PATH.
    """
    import yt_dlp
    from yt_dlp.utils import download_range_func

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "noplaylist": True,
        # "extractor_args": {"youtube": {"player_client": ["web"]}},
    }

    if start_time is not None and duration is not None:
        h, m, s = map(int, start_time.split(":"))
        start_sec = h * 3600 + m * 60 + s
        ydl_opts["download_ranges"] = download_range_func(None, [(start_sec, start_sec + duration)])
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_path


def read_frames(video_path: str | Path) -> Iterator[np.ndarray]:
    """Yields frames (BGR, as OpenCV reads them) one at a time."""
    # TODO: can be replaced with frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def get_video_info(video_path: str | Path) -> dict:
    # TODO: Can be replaced with sv.VideoInfo.from_video_path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter for writing annotated frames back out."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # TODO: consider using sv.VideoWriter instead of this wrapper. Also, sv.VideoSink(target_path, video_info) can be used to write 
        # frames with the same fps/size as the input video, without needing to pass those parameters explicitly.
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
