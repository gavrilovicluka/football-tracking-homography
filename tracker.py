"""
Wraps ByteTrack (via supervision) to assign persistent track IDs
to detections across frames.
"""
import supervision as sv
from collections import defaultdict, deque
import numpy as np


class PlayerTracker:
    def __init__(
        self,
        frame_rate: int = 25,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        min_hits_to_confirm: int = 3,       # a track needs to show up in 3 of the last 5 frames before it's drawn - enough to kill single-frame flukes (a penalty spot briefly flagged as ball) without noticeably delaying real detections.
        history_len: int = 5,
    ):
        """
        lost_track_buffer: how many frames a track survives with no matching
        detection before being dropped - raise this if players getting briefly
        occluded (by other players/ref) are losing their ID too easily.
        """
        self.tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        self.min_hits_to_confirm = min_hits_to_confirm
        self._hit_history = defaultdict(lambda: deque(maxlen=history_len))

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns detections with .tracker_id populated."""
        # return self.tracker.update_with_detections(detections)
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked

        keep_mask = []
        for tid in tracked.tracker_id:
            self._hit_history[tid].append(1)
            keep_mask.append(sum(self._hit_history[tid]) >= self.min_hits_to_confirm)
        return tracked[np.array(keep_mask)]

    # TODO: without usages yet
    def reset(self):
        """Call between separate clips so track IDs don't carry over."""
        self.tracker.reset()
