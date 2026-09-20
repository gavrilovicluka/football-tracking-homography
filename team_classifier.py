"""
Team classification: assigns each tracked player to one of two teams based on
kit appearance, using SigLIP embeddings -> UMAP dimensionality reduction ->
KMeans clustering. Goalkeepers and referees are excluded from clustering -
they're already distinct detection classes, only PLAYER_CLASS_ID gets clustered.

Two-phase usage:
    1. fit(sample_crops)  - call once on a batch of player crops collected across
                             several sample frames, to learn the two team clusters.
    2. predict(crops)     - call per-frame afterward to get a team_id (0 or 1) per crop.

assign_team_ids() applies per-track majority voting on top of predict(), so a given
tracked player's team assignment stabilizes over time instead of flickering frame
to frame on noisy single-frame predictions.
"""
from collections import defaultdict, Counter
from time import perf_counter
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from sklearn.cluster import KMeans
import umap

from config import BATCH_SIZE, SIGLIP_MODEL_NAME

def extract_crops(frame: np.ndarray, xyxy: np.ndarray) -> List[np.ndarray]:
    """Crops out each box from a frame. xyxy: (N, 4) array of [x1, y1, x2, y2]."""

    # TODO: can be replaced with sv.crop_image(frame, xyxy)
    crops = []
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in xyxy.astype(int):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crops.append(frame[y1:y2, x1:x2])
    return crops


class TeamClassifier:
    def __init__(
            self, 
            device: str = "cpu", 
            n_teams: int = 2,
            classification_interval: int | None = None,
    ):
        if device.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is not available."
                )

            self.device = torch.device(
                f"cuda:{device}"
            )
        else:
            self.device = torch.device(device)

        self.n_teams = n_teams

        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.embedding_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device).eval()

        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=n_teams, n_init=10, random_state=42)

        self._fitted = False
        # per-track history of predicted team_id, used for majority-vote smoothing
        self._track_history: dict[int, Counter] = defaultdict(Counter)

        self._track_team_cache: dict[int, int] = {}
        self.min_votes_to_lock = 3

        self.classification_interval = classification_interval

        self._classification_frame_counter = 0
        self._last_team_prediction: dict[int, int] = {}
        self._last_classification_frame: dict[int, int] = {}

    def _extract_embeddings(
            self, 
            crops: List[np.ndarray], 
            batch_size: int = BATCH_SIZE
    ) -> np.ndarray:
        """crops: list of BGR numpy arrays (as read by OpenCV)."""

        all_embeddings = []

        for start in range(0, len(crops), batch_size):
            batch = crops[start:start + batch_size]

            images = [
                Image.fromarray(crop[:, :, ::-1])
                for crop in batch
            ]

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.embedding_model(**inputs)

            embeddings = outputs.last_hidden_state.mean(
                dim=1
            )

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

        return np.concatenate(
            all_embeddings,
            axis=0,
        )

    def fit(self, sample_crops: List[np.ndarray]):
        """Call once on player crops sampled across the clip to learn team clusters."""
        if len(sample_crops) < self.n_teams * 2:
            raise ValueError(
                f"Need more sample crops to fit {self.n_teams} clusters, got {len(sample_crops)}"
            )

        start = perf_counter()
        embeddings = self._extract_embeddings(sample_crops)
        print(
            f"SigLIP embeddings: "
            f"{perf_counter() - start:.2f}s"
        )

        start = perf_counter()
        reduced = self.reducer.fit_transform(embeddings)    # Trains UMAP, and after that runs the projection on the input embeddings. The result is a (N, 3) array of reduced embeddings.
        print(
            f"UMAP fit + transform: "
            f"{perf_counter() - start:.2f}s"
        )

        print(f"Reduced embeddings shape: {reduced.shape} for {len(sample_crops)} crops")

        start = perf_counter()
        self.cluster_model.fit(reduced)
        print(
            f"KMeans fit: "
            f"{perf_counter() - start:.2f}s"
        )

        self._fitted = True

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Returns an array of team_id (0/1) for each crop, in order."""
        if not self._fitted:
            raise RuntimeError("TeamClassifier.fit() must be called before predict()")
        if len(crops) == 0:
            return np.array([], dtype=int)
        embeddings = self._extract_embeddings(crops)
        reduced = self.reducer.transform(embeddings)
        return self.cluster_model.predict(reduced)

    def assign_team_ids(self, tracker_ids: np.ndarray, team_predictions: np.ndarray) -> np.ndarray:
        """Smooths per-frame team predictions with a per-track majority vote,
        so a given player doesn't flicker between teams across frames."""
        stable_ids = []
        for tid, pred in zip(tracker_ids, team_predictions):
            self._track_history[tid][int(pred)] += 1
            stable_team = self._track_history[tid].most_common(1)[0][0]
            stable_ids.append(stable_team)
        return np.array(stable_ids, dtype=int)

    def predict_tracked(
        self,
        crops: list[np.ndarray],
        tracker_ids: np.ndarray,
    ) -> np.ndarray:
        """
        classification_interval=None  → history logic
        classification_interval=1     → SigLIP every frame
        classification_interval=3     → SigLIP every 3rd frame
        """
        self._classification_frame_counter += 1

        # Periodic classification mode
        if self.classification_interval is not None:
            team_ids = np.full(len(crops), -1, dtype=int)
            indices_to_classify = []

            for i, tracker_id in enumerate(tracker_ids):
                tracker_id = int(tracker_id)

                last_frame = self._last_classification_frame.get(tracker_id)

                if (
                    last_frame is None
                    or self._classification_frame_counter - last_frame
                    >= self.classification_interval
                ):
                    indices_to_classify.append(i)
                else:
                    team_ids[i] = self._last_team_prediction[tracker_id]

            if indices_to_classify:
                predictions = self.predict([
                    crops[i] for i in indices_to_classify
                ])

                for index, prediction in zip(
                    indices_to_classify,
                    predictions,
                ):
                    tracker_id = int(tracker_ids[index])
                    prediction = int(prediction)

                    self._last_team_prediction[tracker_id] = prediction
                    self._last_classification_frame[tracker_id] = (
                        self._classification_frame_counter
                    )
                    team_ids[index] = prediction

            return team_ids

        # Existing history-based logic
        return self._predict_with_history(crops, tracker_ids)

    def _predict_with_history(
        self,
        crops: list[np.ndarray],
        tracker_ids: np.ndarray,
    ) -> np.ndarray:
        team_ids = np.full(len(crops), -1, dtype=int)
        
        unresolved_indices = []

        for i, tracker_id in enumerate(tracker_ids):
            tracker_id = int(tracker_id)

            if tracker_id in self._track_team_cache:
                team_ids[i] = self._track_team_cache[tracker_id]
            else:
                unresolved_indices.append(i)

        if not unresolved_indices:
            return team_ids

        unresolved_crops = [
            crops[i]
            for i in unresolved_indices
        ]

        predictions = self.predict(unresolved_crops)

        for index, prediction in zip(
            unresolved_indices,
            predictions,
        ):
            tracker_id = int(tracker_ids[index])
            prediction = int(prediction)

            history = self._track_history[tracker_id]
            history[prediction] += 1

            stable_team = history.most_common(1)[0][0]
            team_ids[index] = stable_team

            if sum(history.values()) >= self.min_votes_to_lock:
                self._track_team_cache[tracker_id] = stable_team

        return team_ids