import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer


class PitchProjector:
    MIN_LANDMARKS = 5
    MIN_INLIERS = 4
    MIN_INLIER_RATIO = 0.6
    MIN_WEIGHTED_INLIER_RATIO = 0.65
    RANSAC_REPROJECTION_THRESHOLD = 12.0
    MAX_PROJECTION_SHIFT_MM = 2500.0

    def __init__(self):
        self.config = SoccerPitchConfiguration()
        self.pitch_vertices = np.asarray(
            self.config.vertices,
            dtype=np.float32,
        )
        self.transformer = None

    def update(
        self,
        landmark_points: np.ndarray,
        landmark_indices: np.ndarray,
        landmark_confidences: np.ndarray | None = None,
        validation_points: np.ndarray | None = None,
    ) -> bool:
        """
        Fits and installs a new image -> 2D pitch transformation when supported.
        """

        source = np.asarray(landmark_points, dtype=np.float32)
        indices = np.asarray(landmark_indices, dtype=int)
        if source.ndim != 2 or source.shape[1:] != (2,) or len(source) != len(indices):
            return False

        if landmark_confidences is None:
            confidences = np.ones(len(source), dtype=np.float32)
        else:
            confidences = np.asarray(landmark_confidences, dtype=np.float32)
            if confidences.shape != (len(source),):
                return False

        valid = (
            np.isfinite(source).all(axis=1)
            & np.isfinite(confidences)
            & (confidences > 0)
            & (indices >= 0)
            & (indices < len(self.pitch_vertices))
        )
        source = source[valid]
        indices = indices[valid]
        confidences = confidences[valid]
        if len(source) < self.MIN_LANDMARKS:
            return False

        target = self.pitch_vertices[indices]
        try:
            _, inlier_mask = cv2.findHomography(
                target,
                source,
                cv2.RANSAC,
                self.RANSAC_REPROJECTION_THRESHOLD,
                maxIters=2000,
                confidence=0.995,
            )
        except cv2.error:
            return False
        if inlier_mask is None:
            return False

        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        minimum_inliers = max(
            self.MIN_INLIERS,
            int(np.ceil(self.MIN_INLIER_RATIO * len(source))),
        )
        if inlier_count < minimum_inliers:
            return False

        weighted_inlier_ratio = confidences[inliers].sum() / confidences.sum()
        if weighted_inlier_ratio < self.MIN_WEIGHTED_INLIER_RATIO:
            return False

        inlier_source = source[inliers]
        inlier_target = target[inliers]
        if any(
            np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
            for points in (inlier_source, inlier_target)
        ):
            return False

        try:
            candidate = ViewTransformer(source=inlier_source, target=inlier_target)
        except (ValueError, cv2.error):
            return False
        if not np.isfinite(candidate.m).all():
            return False

        if self.transformer is not None:
            probes = source if validation_points is None else np.asarray(
                validation_points,
                dtype=np.float32,
            )
            if probes.ndim != 2 or probes.shape[1:] != (2,):
                return False
            probes = probes[np.isfinite(probes).all(axis=1)]
            if len(probes) == 0:
                return False
            try:
                previous_xy = self.transformer.transform_points(probes)
                candidate_xy = candidate.transform_points(probes)
            except (ValueError, cv2.error):
                return False
            shifts = np.linalg.norm(candidate_xy - previous_xy, axis=1)
            if not np.isfinite(shifts).all() or shifts.max() > self.MAX_PROJECTION_SHIFT_MM:
                return False

        self.transformer = candidate

        return True

    def transform_points(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError(
                "Pitch transformation has not been initialized"
            )

        return self.transformer.transform_points(
            points.astype(np.float32)
        )