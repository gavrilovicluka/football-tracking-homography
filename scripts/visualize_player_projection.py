import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from config import PITCH_KEYPOINT_WEIGHTS_PATH, PLAYER_CROPS_SAMPLE_COUNT, WEIGHTS_PATH
from constants import PLAYER_CLASS_ID, TEAM_COLORS
from utils import collect_fitting_crops

import supervision as sv

from detector import FootballDetector
from pitch_projection import PitchProjector
from team_classifier import TeamClassifier, extract_crops
from pitch_landmark_detector import PitchLandmarkDetector

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

def visualize_player_projection(
    video_path: Path,
    pitch_weights_path,
    player_weights_path,
    num_frames: int = 6,
    device: str = "0",
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    pitch_detector = PitchLandmarkDetector(
        weights_path=pitch_weights_path,
        conf=0.5,
        imgsz=960,
        device=device,
    )

    player_detector = FootballDetector(
        player_weights_path,
        conf=0.25,
        device=device,
    )

    print("Fitting team classifier...")

    team_classifier = TeamClassifier(device=device)

    fitting_crops = collect_fitting_crops(
        video_path,
        player_detector,
        n_samples=PLAYER_CROPS_SAMPLE_COUNT,
    )

    team_classifier.fit(fitting_crops)

    print(f"Fitted on {len(fitting_crops)} player crops.")

    projector = PitchProjector()

    results = []

    for frame_idx in frame_indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx),
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # --------------------------
        # 1. Detect pitch landmarks
        # --------------------------

        landmark_xy, landmark_indices, confidences = (
            pitch_detector.detect(frame)
        )

        if len(landmark_xy) < 4:
            print(
                f"Frame {frame_idx}: "
                f"only {len(landmark_xy)} landmarks"
            )
            continue

        # --------------------------
        # 2. Calculate homography
        # --------------------------

        if not projector.update(
            landmark_xy,
            landmark_indices,
        ):
            continue

        # --------------------------
        # 3. Detect players
        # --------------------------

        detections = player_detector.detect(frame)

        player_mask = (
            detections.class_id == PLAYER_CLASS_ID
        )

        player_detections = detections[
            player_mask
        ]

        # --------------------------
        # 3. Classify players
        # --------------------------
        if len(player_detections) == 0:
            continue

        player_crops = extract_crops(
            frame,
            player_detections.xyxy,
        )

        team_ids = team_classifier.predict(
            player_crops
        )

        # --------------------------
        # 4. Bottom-center of boxes
        # --------------------------

        player_image_points = player_detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        if len(player_image_points) == 0:
            continue

        # --------------------------
        # 5. Project to pitch
        # --------------------------

        player_pitch_points = (
            projector.transform_points(
                player_image_points
            )
        )

        # --------------------------
        # 6. Draw 2D pitch
        # --------------------------

        pitch = draw_pitch(
            projector.config
        )
        
        for team_id in [0, 1]:
            mask = team_ids == team_id

            if not mask.any():
                continue

            pitch = draw_points_on_pitch(
                config=projector.config,
                xy=player_pitch_points[mask],
                face_color=TEAM_COLORS[team_id],
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=pitch,
            )

        original_frame = frame.copy()

        for x, y in player_image_points:
            cv2.circle(
                original_frame,
                (int(x), int(y)),
                6,
                (0, 0, 255),
                -1,
            )

        results.append({
            "frame_idx": int(frame_idx),
            "time_sec": frame_idx / fps,
            "original_frame": original_frame,
            "pitch": pitch,
            "player_image_points": player_image_points,
            "player_pitch_points": player_pitch_points,
            "team_ids": team_ids,
        })

    cap.release()

    for result in results:
        print(
            f"Frame {result['frame_idx']} "
            f"({result['time_sec']:.2f}s)"
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(18, 7),
        )

        # OpenCV frame is BGR, matplotlib expects RGB
        original_rgb = cv2.cvtColor(
            result["original_frame"],
            cv2.COLOR_BGR2RGB,
        )

        pitch_rgb = cv2.cvtColor(
            result["pitch"],
            cv2.COLOR_BGR2RGB,
        )

        axes[0].imshow(original_rgb)
        axes[0].set_title("Original frame")
        axes[0].axis("off")

        axes[1].imshow(pitch_rgb)
        axes[1].set_title("2D pitch projection")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize detected pitch landmarks."
    )

    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu', '0' for GPU 0, etc.",
    )

    args = parser.parse_args()

    visualize_player_projection(
        video_path=args.video_path,
        pitch_weights_path=PITCH_KEYPOINT_WEIGHTS_PATH,
        player_weights_path=WEIGHTS_PATH,
        num_frames=1,
        device=args.device,
    )

if __name__ == "__main__":
    main()