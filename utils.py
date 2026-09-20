from pathlib import Path
import numpy as np

from config import PLAYER_CROPS_SAMPLE_COUNT
from constants import PLAYER_CLASS_ID
from video_io import get_video_info

import supervision as sv
from tqdm import tqdm

from detector import FootballDetector


# We need to collect a sufficient set of player crops to use to train team classification model
def collect_fitting_crops(
        video_path: Path, 
        detector: FootballDetector, 
        n_samples: int = PLAYER_CROPS_SAMPLE_COUNT
) -> list:
    """Runs detection (no tracking needed) on evenly-spaced sample frames across
    the clip, collecting player crops to fit the team classifier on."""
    info = get_video_info(video_path)
    sample_indices = set(
        np.linspace(
            0, 
            info["frame_count"] - 1, 
            n_samples, 
            dtype=int
        )
    )

    frame_generator = sv.get_video_frames_generator(
        source_path=video_path
    )

    crops = []
    # for i, frame in enumerate(read_frames(video_path)):
    for i, frame in enumerate(tqdm(frame_generator, desc="Collecting player crops")):
        if i not in sample_indices:
            continue
        detections = detector.detect(frame)
        detections = detections[
            detections.class_id == PLAYER_CLASS_ID
        ]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    return crops