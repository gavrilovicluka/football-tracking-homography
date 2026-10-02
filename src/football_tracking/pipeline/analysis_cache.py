"""Persistent cache for per-frame analysis results."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import supervision as sv

from football_tracking.config import (
    BATCH_SIZE,
    CLASSIFICATION_INTERVAL,
    IMAGE_SIZE,
    OUTPUTS_DIR,
    PITCH_IMAGE_SIZE,
    PITCH_KEYPOINT_WEIGHTS_PATH,
    PLAYER_CROPS_SAMPLE_COUNT,
    SIGLIP_MODEL_NAME,
    WEIGHTS_PATH,
)
from football_tracking.pipeline.analyzer import FrameResult


CACHE_VERSION = 5


def _file_signature(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def create_frame_analysis_cache(
    video_path: Path,
    frame_count: int,
    fps: float,
    conf: float,
    device: str,
    with_pitch: bool,
    pitch_conf: float,
    pitch_imgsz: int,
    pitch_detect_interval: int,
    ball_conf: float = 0.15,
) -> FrameAnalysisCache:
    from football_tracking.schema import DEFAULT_CLASS_CONF_THRESHOLDS

    metadata = {
        "version": CACHE_VERSION,
        "video": _file_signature(video_path),
        "player_weights": _file_signature(WEIGHTS_PATH),
        "pitch_weights": _file_signature(PITCH_KEYPOINT_WEIGHTS_PATH) if with_pitch else None,
        "frame_count": frame_count,
        "fps": fps,
        "conf": conf,
        "ball_conf": ball_conf,
        "device": device,
        "with_pitch": with_pitch,
        "pitch_conf": pitch_conf,
        "pitch_imgsz": pitch_imgsz,
        "pitch_detect_interval": pitch_detect_interval,
        "player_imgsz": IMAGE_SIZE,
        "class_conf_thresholds": sorted(DEFAULT_CLASS_CONF_THRESHOLDS.items()),
        "classification_interval": CLASSIFICATION_INTERVAL,
        "player_crops_sample_count": PLAYER_CROPS_SAMPLE_COUNT,
        "siglip_model": SIGLIP_MODEL_NAME,
        "embedding_batch_size": BATCH_SIZE,
    }
    serialized_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized_metadata.encode("utf-8")).hexdigest()
    cache_path = OUTPUTS_DIR / ".cache" / "frame-analysis" / f"{cache_key}_v{CACHE_VERSION}.pkl"
    return FrameAnalysisCache(cache_path, cache_key, frame_count)


class FrameAnalysisCache:
    def __init__(self, path: Path, cache_key: str, expected_frame_count: int):
        self.path = path
        self.cache_key = cache_key
        self.expected_frame_count = expected_frame_count
        self.results: list[FrameResult] = []
        self._pending_results: list[FrameResult] = []
        self.hit = self._load()

    def _load(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            with self.path.open("rb") as cache_file:
                payload = pickle.load(cache_file)
            if (
                payload.get("version") != CACHE_VERSION
                or payload.get("key") != self.cache_key
                or len(payload.get("frames", [])) != self.expected_frame_count
            ):
                return False
            self.results = [_restore_result(frame) for frame in payload["frames"]]
            return True
        except Exception as error:
            print(f"Ignoring unreadable analysis cache {self.path}: {error}")
            return False

    def record(self, result: FrameResult) -> None:
        self._pending_results.append(result)
        self.results.append(result)

    def save(self) -> None:
        if len(self._pending_results) != self.expected_frame_count:
            print("Skipping analysis cache: processed frame count did not match video metadata.")
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.path.parent, prefix="analysis-", suffix=".tmp", delete=False
            ) as cache_file:
                temporary_path = Path(cache_file.name)
                pickle.dump(
                    {
                        "version": CACHE_VERSION,
                        "key": self.cache_key,
                        "frames": [_serialize_result(result) for result in self._pending_results],
                    },
                    cache_file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary_path, self.path)
            print(f"Saved frame-analysis cache: {self.path}")
        except Exception as error:
            print(f"Could not save frame-analysis cache: {error}")
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _serialize_result(result: FrameResult) -> dict:
    detections = result.detections
    return {
        "xyxy": detections.xyxy,
        "mask": detections.mask,
        "confidence": detections.confidence,
        "class_id": detections.class_id,
        "tracker_id": detections.tracker_id,
        "team_ids": result.team_ids,
        "player_mask": result.player_mask,
        "landmarks_xy": result.landmarks_xy,
        "pitch_xy": result.pitch_xy,
        "projection_class_ids": result.projection_class_ids,
        "pitch_transform": result.pitch_transform,
    }


def _restore_result(frame: dict) -> FrameResult:
    detections = sv.Detections(
        xyxy=frame["xyxy"],
        mask=frame["mask"],
        confidence=frame["confidence"],
        class_id=frame["class_id"],
        tracker_id=frame["tracker_id"],
    )
    return FrameResult(
        detections=detections,
        team_ids=frame["team_ids"],
        player_mask=frame["player_mask"],
        landmarks_xy=frame["landmarks_xy"],
        pitch_xy=frame["pitch_xy"],
        projection_class_ids=frame["projection_class_ids"],
        pitch_transform=frame.get("pitch_transform"),
    )