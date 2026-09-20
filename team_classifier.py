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
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from sklearn.cluster import KMeans
import umap

from config import SIGLIP_MODEL_NAME

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
    def __init__(self, device: str = "cpu", n_teams: int = 2):
        self.device = device
        self.n_teams = n_teams

        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.embedding_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()

        self.reducer = umap.UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=n_teams, n_init=10, random_state=42)

        self._fitted = False
        # per-track history of predicted team_id, used for majority-vote smoothing
        self._track_history: dict[int, Counter] = defaultdict(Counter)

    def _extract_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """crops: list of BGR numpy arrays (as read by OpenCV)."""

        # TODO: can be implemented with batch processing
        images = [Image.fromarray(crop[:, :, ::-1]) for crop in crops]  # BGR -> RGB
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        # mean-pool patch embeddings into a single vector per crop
        embeddings = outputs.last_hidden_state.mean(dim=1)
        print(f"Embeddings shape: {embeddings.shape} for {len(crops)} crops")

        return embeddings.cpu().numpy()

    def fit(self, sample_crops: List[np.ndarray]):
        """Call once on player crops sampled across the clip to learn team clusters."""
        if len(sample_crops) < self.n_teams * 2:
            raise ValueError(
                f"Need more sample crops to fit {self.n_teams} clusters, got {len(sample_crops)}"
            )
        embeddings = self._extract_embeddings(sample_crops)
        reduced = self.reducer.fit_transform(embeddings)    # Trains UMAP, and after that runs the projection on the input embeddings. The result is a (N, 3) array of reduced embeddings.
        print(f"Reduced embeddings shape: {reduced.shape} for {len(sample_crops)} crops")

        self.cluster_model.fit(reduced)
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