import argparse
from pathlib import Path
import numpy as np
import cv2

import supervision as sv

from config import PITCH_KEYPOINT_WEIGHTS_PATH
from pitch_landmark_detector import PitchLandmarkDetector

from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer

def visualize_pitch_landmarks(
    video_path: Path,
    weights_path,
    num_frames: int = 6,
    device: str = "0",
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    pitch_detector = PitchLandmarkDetector(
        weights_path=weights_path,
        conf=0.5,
        imgsz=960,
        device=device,
    )

    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF1493"),
        radius=8,
    )

    results = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

        ret, frame = cap.read()

        if not ret:
            continue

        keypoints_xy, landmark_indices, confidences = (
            pitch_detector.detect(frame)
        )


        if len(keypoints_xy) >= 4:

            CONFIG = SoccerPitchConfiguration()

            pitch_vertices = np.array(
                CONFIG.vertices,
                dtype=np.float32,
            )

            source_points = keypoints_xy.astype(np.float32)

            target_points = pitch_vertices[
                landmark_indices
            ]

            transformer = ViewTransformer(
                source=source_points,
                target=target_points,
            )

            projected_points = transformer.transform_points(
                source_points
            )

            pitch = draw_pitch(CONFIG)

            pitch = draw_points_on_pitch(
                config=CONFIG,
                xy=projected_points,
                face_color=sv.Color.RED,
                pitch=pitch,
            )
            results.append({
                        "frame_idx": int(frame_idx),
                        "time_sec": frame_idx / fps,
                        "frame": pitch,
                        "keypoints_xy": keypoints_xy,
                        "landmark_indices": landmark_indices,
                        "confidences": confidences,
                    })


        # key_points = sv.KeyPoints(
        #     xy=keypoints_xy[np.newaxis, ...],
        #     keypoint_confidence=confidences[np.newaxis, ...],
        # )

        # annotated_frame = vertex_annotator.annotate(
        #     scene=frame.copy(),
        #     key_points=key_points,
        # )

        # results.append({
        #     "frame_idx": int(frame_idx),
        #     "time_sec": frame_idx / fps,
        #     "frame": annotated_frame,
        #     "keypoints_xy": keypoints_xy,
        #     "landmark_indices": landmark_indices,
        #     "confidences": confidences,
        # })

    cap.release()

    for result in results:
        print(
            f"Frame {result['frame_idx']} "
            f"({result['time_sec']:.2f}s)"
        )

        print("Landmarks:", result["landmark_indices"])

        sv.plot_image(result["frame"])

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

    visualize_pitch_landmarks(
        video_path=args.video_path,
        weights_path=PITCH_KEYPOINT_WEIGHTS_PATH,
        num_frames=args.num_frames,
        device=args.device,
    )

if __name__ == "__main__":
    main()