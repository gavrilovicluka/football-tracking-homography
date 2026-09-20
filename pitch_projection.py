import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer


class PitchProjector:
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
    ) -> bool:
        """
        Builds a new image -> 2D pitch transformation.
        """

        if len(landmark_points) < 4:
            return False

        source = landmark_points.astype(np.float32)

        target = self.pitch_vertices[
            landmark_indices
        ].astype(np.float32)

        self.transformer = ViewTransformer(
            source=source,
            target=target,
        )

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