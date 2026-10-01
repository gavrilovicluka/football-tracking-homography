"""
Offline ball tracking. Picks at most one ball detection per frame by motion
consistency across the whole clip, then fills short gaps by interpolation.

Steps:
1. Merge duplicate boxes on the same object.
2. Build short tracklets by associating detections frame to frame.
3. Choose the chain of tracklets that best forms one plausible ball path.
4. Interpolate short gaps between accepted positions.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    xyxy: np.ndarray
    confidence: float

    @property
    def center(self) -> np.ndarray:
        return (self.xyxy[:2] + self.xyxy[2:]) / 2


@dataclass(frozen=True)
class BallTrackPoint:
    xyxy: np.ndarray
    confidence: float | None
    interpolated: bool

    @property
    def center(self) -> np.ndarray:
        return (self.xyxy[:2] + self.xyxy[2:]) / 2


@dataclass
class _Tracklet:
    frames: list[int] = field(default_factory=list)
    candidates: list[BallCandidate] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    @property
    def score(self) -> float:
        return sum(candidate.confidence for candidate in self.candidates)

    def predict(self, frame: int) -> np.ndarray:
        last = self.candidates[-1].center
        if len(self.frames) < 2:
            return last
        first_index = max(0, len(self.frames) - 3)
        elapsed = self.frames[-1] - self.frames[first_index]
        velocity = (last - self.candidates[first_index].center) / elapsed
        return last + velocity * (frame - self.frames[-1])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class BallTracker:
    def __init__(
        self,
        duplicate_distance: float = 10.0,
        duplicate_iou: float = 0.3,
        gate_base: float = 15.0,
        gate_per_frame: float = 25.0,
        gate_max: float = 90.0,
        max_tracklet_gap: int = 5,
        link_base: float = 10.0,
        link_speed: float = 30.0,
        max_link_gap: int = 50,
        link_weight: float = 1.5,
        jump_penalty: float = 1.5,
        max_interpolation_gap: int = 3,
    ):
        """
        Distances are in pixels, gaps in frames.

        gate_*: how far a detection may be from a tracklet's predicted position.
        link_*: how far apart two tracklets may be to count as the same ball path.
        link_weight: cost of a link at the maximum allowed distance; shorter links
            cost proportionally less, so detours through false boxes are rejected.
        jump_penalty: cost of an unexplained jump between tracklets; a tracklet
            whose summed confidence is below it is dropped unless it connects to
            the path, which removes short isolated false positives.
        max_interpolation_gap: longer gaps are left empty.
        """
        self.duplicate_distance = duplicate_distance
        self.duplicate_iou = duplicate_iou
        self.gate_base = gate_base
        self.gate_per_frame = gate_per_frame
        self.gate_max = gate_max
        self.max_tracklet_gap = max_tracklet_gap
        self.link_base = link_base
        self.link_speed = link_speed
        self.max_link_gap = max_link_gap
        self.link_weight = link_weight
        self.jump_penalty = jump_penalty
        self.max_interpolation_gap = max_interpolation_gap

    def track(
        self, candidates_per_frame: list[list[BallCandidate]]
    ) -> list[BallTrackPoint | None]:
        """Returns one entry per frame: the ball position, or None if unknown."""
        merged = [self._merge_duplicates(candidates) for candidates in candidates_per_frame]
        tracklets = self._build_tracklets(merged)
        chain = self._select_chain(tracklets)
        return self._fill(chain, len(candidates_per_frame))

    def _merge_duplicates(self, candidates: list[BallCandidate]) -> list[BallCandidate]:
        kept: list[BallCandidate] = []
        for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            if not any(
                np.linalg.norm(candidate.center - other.center) <= self.duplicate_distance
                or _iou(candidate.xyxy, other.xyxy) >= self.duplicate_iou
                for other in kept
            ):
                kept.append(candidate)
        return kept

    def _gate(self, frames_elapsed: int) -> float:
        return min(self.gate_base + self.gate_per_frame * frames_elapsed, self.gate_max)

    def _build_tracklets(self, merged: list[list[BallCandidate]]) -> list[_Tracklet]:
        active: list[_Tracklet] = []
        finished: list[_Tracklet] = []

        for frame, candidates in enumerate(merged):
            still_active = []
            for tracklet in active:
                if frame - tracklet.end > self.max_tracklet_gap + 1:
                    finished.append(tracklet)
                else:
                    still_active.append(tracklet)
            active = still_active

            pairs = []
            for tracklet_index, tracklet in enumerate(active):
                elapsed = frame - tracklet.end
                predicted = tracklet.predict(frame)
                last = tracklet.candidates[-1].center
                gate = self._gate(elapsed)
                for candidate_index, candidate in enumerate(candidates):
                    # The last position also counts so sudden kicks are not rejected.
                    distance = min(
                        np.linalg.norm(candidate.center - predicted),
                        np.linalg.norm(candidate.center - last),
                    )
                    if distance <= gate:
                        pairs.append((distance, tracklet_index, candidate_index))

            used_tracklets: set[int] = set()
            used_candidates: set[int] = set()
            for _, tracklet_index, candidate_index in sorted(pairs):
                if tracklet_index in used_tracklets or candidate_index in used_candidates:
                    continue
                active[tracklet_index].frames.append(frame)
                active[tracklet_index].candidates.append(candidates[candidate_index])
                used_tracklets.add(tracklet_index)
                used_candidates.add(candidate_index)

            for candidate_index, candidate in enumerate(candidates):
                if candidate_index not in used_candidates:
                    active.append(_Tracklet(frames=[frame], candidates=[candidate]))

        return finished + active

    def _link_ratio(self, earlier: _Tracklet, later: _Tracklet) -> float:
        """Jump distance relative to the allowed distance; above 1 means not linkable."""
        gap = later.start - earlier.end
        if gap <= 0 or gap > self.max_link_gap:
            return float("inf")
        distance = np.linalg.norm(later.candidates[0].center - earlier.candidates[-1].center)
        return distance / (self.link_base + self.link_speed * gap)

    def _linkable(self, earlier: _Tracklet, later: _Tracklet) -> bool:
        return self._link_ratio(earlier, later) <= 1.0

    def _transition_cost(self, earlier: _Tracklet, later: _Tracklet) -> float:
        ratio = self._link_ratio(earlier, later)
        return self.link_weight * ratio if ratio <= 1.0 else self.jump_penalty

    def _select_chain(self, tracklets: list[_Tracklet]) -> list[_Tracklet]:
        if not tracklets:
            return []
        ordered = sorted(tracklets, key=lambda t: (t.end, t.start))
        best = [0.0] * len(ordered)
        previous = [-1] * len(ordered)

        for j, tracklet in enumerate(ordered):
            best[j] = tracklet.score
            for i in range(j):
                if ordered[i].end >= tracklet.start:
                    continue
                value = best[i] + tracklet.score - self._transition_cost(ordered[i], tracklet)
                if value > best[j]:
                    best[j] = value
                    previous[j] = i

        index = int(np.argmax(best))
        chain = []
        while index != -1:
            chain.append(ordered[index])
            index = previous[index]
        return chain[::-1]

    def _fill(self, chain: list[_Tracklet], frame_count: int) -> list[BallTrackPoint | None]:
        track: list[BallTrackPoint | None] = [None] * frame_count
        for tracklet in chain:
            for frame, candidate in zip(tracklet.frames, tracklet.candidates):
                track[frame] = BallTrackPoint(candidate.xyxy, candidate.confidence, interpolated=False)

        for tracklet_index, tracklet in enumerate(chain):
            segments = list(zip(tracklet.frames, tracklet.candidates))
            if tracklet_index + 1 < len(chain):
                following = chain[tracklet_index + 1]
                if self._linkable(tracklet, following):
                    segments.append((following.start, following.candidates[0]))
            for (frame_a, a), (frame_b, b) in zip(segments, segments[1:]):
                gap = frame_b - frame_a - 1
                if gap < 1 or gap > self.max_interpolation_gap:
                    continue
                for frame in range(frame_a + 1, frame_b):
                    t = (frame - frame_a) / (frame_b - frame_a)
                    track[frame] = BallTrackPoint(
                        a.xyxy + (b.xyxy - a.xyxy) * t, confidence=None, interpolated=True
                    )
        return track
