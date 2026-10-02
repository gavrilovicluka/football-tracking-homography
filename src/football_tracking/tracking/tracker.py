"""
Wraps ByteTrack (via supervision) to assign persistent track IDs
to detections across frames.
"""
import supervision as sv
from collections import defaultdict, deque
import numpy as np

from football_tracking.schema import PLAYER_CLASS_ID, REFEREE_CLASS_ID


class PlayerTracker:
    CROSS_CLASS_DUPLICATE_IOU_THRESHOLD = 0.8
    PREVIOUS_TRACK_IOU_THRESHOLD = 0.5

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
        self._stable_class_by_id: dict[int, int] = {}
        self._pending_class_by_id: dict[int, tuple[int, int]] = {}
        self._last_box_by_id: dict[int, np.ndarray] = {}

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Returns detections with .tracker_id populated."""
        detections = self._filter_cross_class_duplicates(detections)
        tracked = self.tracker.update_with_detections(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked

        keep_mask = []
        for tid in tracked.tracker_id:
            self._hit_history[tid].append(1)
            keep_mask.append(sum(self._hit_history[tid]) >= self.min_hits_to_confirm)
        tracked = tracked[np.array(keep_mask)]
        self._stabilize_player_referee_classes(tracked)
        for tracker_id, box in zip(tracked.tracker_id, tracked.xyxy):
            track_id = int(tracker_id)
            if track_id in self._stable_class_by_id:
                self._last_box_by_id[track_id] = box.copy()
        return tracked

    def _filter_cross_class_duplicates(self, detections: sv.Detections) -> sv.Detections:
        if (
            len(detections) < 2
            or detections.class_id is None
        ):
            return detections

        candidate_indices = np.flatnonzero(
            np.isin(detections.class_id, [PLAYER_CLASS_ID, REFEREE_CLASS_ID])
        )
        if len(candidate_indices) < 2:
            return detections

        confidences = detections.confidence

        def priority(index: int) -> tuple[int, int, float]:
            previous_class, previous_iou = self._previous_class_for_box(
                detections.xyxy[index]
            )
            confidence = (
                float(confidences[index])
                if confidences is not None
                else 0.0
            )
            return (
                confidence,
                int(previous_class == detections.class_id[index]),
                int(previous_iou * 1000),
            )

        ordered_indices = sorted(
            candidate_indices.tolist(),
            key=priority,
            reverse=True,
        )
        keep_mask = np.ones(len(detections), dtype=bool)
        for order_index, kept_index in enumerate(ordered_indices):
            if not keep_mask[kept_index]:
                continue
            for candidate_index in ordered_indices[order_index + 1:]:
                if not keep_mask[candidate_index]:
                    continue
                if detections.class_id[kept_index] == detections.class_id[candidate_index]:
                    continue
                if self._box_iou(
                    detections.xyxy[kept_index],
                    detections.xyxy[candidate_index],
                ) >= self.CROSS_CLASS_DUPLICATE_IOU_THRESHOLD:
                    keep_mask[candidate_index] = False

        return detections[keep_mask]

    def _previous_class_for_box(self, box: np.ndarray) -> tuple[int | None, float]:
        best_class = None
        best_iou = 0.0
        for track_id, previous_box in self._last_box_by_id.items():
            track_class = self._stable_class_by_id.get(track_id)
            if track_class not in {PLAYER_CLASS_ID, REFEREE_CLASS_ID}:
                continue
            overlap = self._box_iou(box, previous_box)
            if overlap > best_iou:
                best_class = track_class
                best_iou = overlap

        if best_iou < self.PREVIOUS_TRACK_IOU_THRESHOLD:
            return None, 0.0
        return best_class, best_iou

    @staticmethod
    def _box_iou(first_box: np.ndarray, second_box: np.ndarray) -> float:
        intersection_top_left = np.maximum(first_box[:2], second_box[:2])
        intersection_bottom_right = np.minimum(first_box[2:], second_box[2:])
        intersection_size = np.maximum(
            intersection_bottom_right - intersection_top_left,
            0,
        )
        intersection_area = float(np.prod(intersection_size))
        first_size = np.maximum(first_box[2:] - first_box[:2], 0)
        second_size = np.maximum(second_box[2:] - second_box[:2], 0)
        union_area = (
            float(np.prod(first_size))
            + float(np.prod(second_size))
            - intersection_area
        )
        return intersection_area / union_area if union_area > 0 else 0.0

    def _stabilize_player_referee_classes(self, detections: sv.Detections) -> None:
        if detections.class_id is None or detections.tracker_id is None:
            return

        detections.class_id = detections.class_id.copy()
        valid_classes = {PLAYER_CLASS_ID, REFEREE_CLASS_ID}
        for index, (tracker_id, observed_class) in enumerate(
            zip(detections.tracker_id, detections.class_id)
        ):
            if tracker_id is None or tracker_id == -1:
                continue

            track_id = int(tracker_id)
            observed_class = int(observed_class)
            if observed_class not in valid_classes:
                self._stable_class_by_id.pop(track_id, None)
                self._pending_class_by_id.pop(track_id, None)
                continue

            stable_class = self._stable_class_by_id.get(track_id)
            if stable_class is None:
                self._stable_class_by_id[track_id] = observed_class
                continue

            if observed_class == stable_class:
                self._pending_class_by_id.pop(track_id, None)
                continue

            pending_class, pending_count = self._pending_class_by_id.get(
                track_id, (observed_class, 0)
            )
            pending_count = pending_count + 1 if pending_class == observed_class else 1
            if pending_count >= 2:
                stable_class = observed_class
                self._stable_class_by_id[track_id] = stable_class
                self._pending_class_by_id.pop(track_id, None)
            else:
                self._pending_class_by_id[track_id] = (observed_class, pending_count)

            detections.class_id[index] = stable_class

    # TODO: without usages yet
    def reset(self):
        """Call between separate clips so track IDs don't carry over."""
        self.tracker.reset()
        self._hit_history.clear()
        self._stable_class_by_id.clear()
        self._pending_class_by_id.clear()
        self._last_box_by_id.clear()
