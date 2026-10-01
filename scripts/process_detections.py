import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from football_tracking.media.video import VideoWriter, get_video_info, read_frames
from football_tracking.schema import BALL_CLASS_ID, PLAYER_CLASS_ID
from football_tracking.detection.players import FootballDetector
from football_tracking.tracking.tracker import PlayerTracker
from football_tracking.config import WEIGHTS_PATH
import supervision as sv

def process_detections(
        video_path: Path, 
        output_path: Path, 
        conf: float = 0.25, 
        device: str = "cpu"
):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    # Initialize only the detector and tracker (no team fitting)
    detector = FootballDetector(WEIGHTS_PATH, conf=conf, device=device)
    tracker = PlayerTracker(frame_rate=max(1, round(info["fps"])))
    tracker.reset()

    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])
    
    # Initialize basic Supervision annotators (e.g., ellipses and triangles for the ball)
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    triangle_annotator = sv.TriangleAnnotator(
        base=20, 
        height=15, 
        color=sv.Color.from_hex("#FFFF00"),
    )

    unique_ids = set()

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing Track-Only")
        ):
            # Detection
            detections = detector.detect(frame)

            # Adding padding to the ball detections to ensure they are fully captured and not lost due to small bounding boxes.
            ball_mask = detections.class_id == BALL_CLASS_ID
            if ball_mask.any():
                detections.xyxy[ball_mask] = sv.pad_boxes(
                    xyxy=detections.xyxy[ball_mask], px=10
                )

            # Split detections into players/referees and ball for tracking. 
            # The ball is usually too fast and small to track reliably, so only track players and referees.
            non_ball_mask = detections.class_id != BALL_CLASS_ID
            non_ball_detections = detections[non_ball_mask]
            
            non_ball_detections = tracker.update(non_ball_detections)

            # Extract the ball detections and assign them a tracker_id of -1 to ensure they are not tracked.
            ball_detections = detections[ball_mask]
            if len(ball_detections) > 0:
                ball_detections.tracker_id = np.full(len(ball_detections), -1, dtype=int)
            
            # Merge the detections back together
            detections = sv.Detections.merge([non_ball_detections, ball_detections])

            if detections.tracker_id is not None:
                # Filtering out tracker IDs for players only (where ID is not None/-1)
                valid_ids = detections.tracker_id[detections.tracker_id != -1]
                unique_ids.update(valid_ids.tolist())

            # Visualization (annotation) without team classification
            annotated_frame = frame.copy()
            
            # Split drawing into players/referees and ball for different annotation styles
            players_and_refs = detections[detections.class_id != BALL_CLASS_ID]
            balls = detections[detections.class_id == BALL_CLASS_ID]

            if len(players_and_refs) > 0:
                annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=players_and_refs)
                labels = [f"ID: {tracker_id}" for tracker_id in players_and_refs.tracker_id]
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=players_and_refs, labels=labels)

            if len(balls) > 0:
                annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=balls)

            writer.write(annotated_frame)

    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process and annotate detected objects in a video."
    )

    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    process_detections(
        video_path=args.video_path,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()