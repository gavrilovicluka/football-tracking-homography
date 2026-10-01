import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from football_tracking.config import WEIGHTS_PATH
from football_tracking.detection.players import FootballDetector
from football_tracking.media.video import VideoWriter, get_video_info, read_frames
from football_tracking.schema import BALL_CLASS_ID, DEFAULT_CLASS_CONF_THRESHOLDS
from football_tracking.tracking.ball import BallCandidate, BallTracker, BallTrackPoint
from football_tracking.tracking.tracker import PlayerTracker

DETECTED_COLOR = (0, 220, 0)
INTERPOLATED_COLOR = (0, 165, 255)
CANDIDATE_COLOR = (200, 200, 200)
TRAIL_LENGTH = 15
VIEWER_WINDOW = "Ball tracking"


def process_detections(
        video_path: Path,
        output_path: Path,
        ball_conf: float = 0.15,
        max_interpolation_gap: int = 3,
        device: str = "cpu",
):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    class_conf_thresholds = {**DEFAULT_CLASS_CONF_THRESHOLDS, BALL_CLASS_ID: ball_conf}
    detector = FootballDetector(
        WEIGHTS_PATH,
        conf=min(class_conf_thresholds.values()),
        device=device,
        class_conf_thresholds=class_conf_thresholds,
    )
    tracker = PlayerTracker(frame_rate=max(1, round(info["fps"])))
    tracker.reset()

    # Pass 1: detect everything; the ball is tracked afterwards using the whole clip.
    players_per_frame: list[sv.Detections] = []
    candidates_per_frame: list[list[BallCandidate]] = []
    for frame in tqdm(read_frames(video_path), total=info["frame_count"], desc="Detecting"):
        detections = detector.detect(frame)
        ball_detections = detections[detections.class_id == BALL_CLASS_ID]
        candidates_per_frame.append([
            BallCandidate(xyxy=box.astype(float), confidence=float(confidence))
            for box, confidence in zip(ball_detections.xyxy, ball_detections.confidence)
        ])
        players_per_frame.append(tracker.update(detections[detections.class_id != BALL_CLASS_ID]))

    ball_track = BallTracker(max_interpolation_gap=max_interpolation_gap).track(candidates_per_frame)

    track_csv_path = output_path.with_name(f"{output_path.stem}_ball_track.csv")
    write_track_csv(track_csv_path, ball_track, candidates_per_frame, info["fps"])

    # Pass 2: render.
    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])
    ellipse_annotator = sv.EllipseAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    unique_ids = set()
    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=len(ball_track), desc="Rendering")
        ):
            if frame_idx >= len(ball_track):
                break
            annotated_frame = frame.copy()

            players = players_per_frame[frame_idx]
            if len(players) > 0:
                unique_ids.update(players.tracker_id.tolist())
                annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=players)
                labels = [f"ID: {tracker_id}" for tracker_id in players.tracker_id]
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame, detections=players, labels=labels
                )

            draw_ball_overlay(
                annotated_frame, frame_idx, ball_track, candidates_per_frame[frame_idx], len(ball_track)
            )
            writer.write(annotated_frame)
    finally:
        writer.release()

    detected = sum(point is not None and not point.interpolated for point in ball_track)
    interpolated = sum(point is not None and point.interpolated for point in ball_track)
    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Ball track CSV: {track_csv_path}")
    print(
        f"Ball: detected in {detected}, interpolated in {interpolated}, "
        f"missing in {len(ball_track) - detected - interpolated} of {len(ball_track)} frames"
    )
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")


def write_track_csv(
        path: Path,
        ball_track: list[BallTrackPoint | None],
        candidates_per_frame: list[list[BallCandidate]],
        fps: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_index", "time_seconds", "status", "confidence",
            "x1", "y1", "x2", "y2", "center_x", "center_y", "candidate_count",
        ])
        for frame_idx, (point, candidates) in enumerate(zip(ball_track, candidates_per_frame)):
            time_seconds = f"{frame_idx / fps:.3f}" if fps > 0 else ""
            if point is None:
                writer.writerow([frame_idx, time_seconds, "none", "", "", "", "", "", "", "", len(candidates)])
                continue
            x1, y1, x2, y2 = point.xyxy
            center_x, center_y = point.center
            writer.writerow([
                frame_idx, time_seconds,
                "interpolated" if point.interpolated else "detected",
                "" if point.confidence is None else f"{point.confidence:.4f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                f"{center_x:.1f}", f"{center_y:.1f}", len(candidates),
            ])


def draw_ball_overlay(
        frame: np.ndarray,
        frame_idx: int,
        ball_track: list[BallTrackPoint | None],
        candidates: list[BallCandidate],
        frame_count: int,
):
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CANDIDATE_COLOR, 1)
        cv2.putText(
            frame, f"{candidate.confidence:.2f}", (x1, y2 + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, CANDIDATE_COLOR, 1, cv2.LINE_AA,
        )

    trail = ball_track[max(0, frame_idx - TRAIL_LENGTH):frame_idx + 1]
    for previous, current in zip(trail, trail[1:]):
        if previous is None or current is None:
            continue
        color = INTERPOLATED_COLOR if current.interpolated else DETECTED_COLOR
        cv2.line(
            frame, tuple(previous.center.astype(int)), tuple(current.center.astype(int)),
            color, 2, cv2.LINE_AA,
        )

    point = ball_track[frame_idx]
    if point is None:
        status = "ball: none"
        color = (0, 0, 255)
    else:
        color = INTERPOLATED_COLOR if point.interpolated else DETECTED_COLOR
        status = "ball: interpolated" if point.interpolated else f"ball: detected {point.confidence:.2f}"
        center_x = int(point.center[0])
        top = int(point.xyxy[1]) - 6
        triangle = np.array([[center_x, top], [center_x - 10, top - 16], [center_x + 10, top - 16]])
        cv2.fillPoly(frame, [triangle], color)
        cv2.polylines(frame, [triangle], True, (0, 0, 0), 1, cv2.LINE_AA)

    header = f"Frame {frame_idx}/{frame_count - 1} | {status} | candidates: {len(candidates)}"
    cv2.rectangle(frame, (0, 0), (620, 34), (0, 0, 0), -1)
    cv2.putText(frame, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def view_video(video_path: Path):
    """Frame-by-frame viewer: a/d or arrows step, space plays, slider jumps, q/Esc quits."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    state = {"index": 0, "playing": False, "suppress": False}

    def on_trackbar(value):
        if not state["suppress"]:
            state["index"] = value
            state["playing"] = False

    cv2.namedWindow(VIEWER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIEWER_WINDOW, 1280, 760)
    cv2.createTrackbar("Frame", VIEWER_WINDOW, 0, max(frame_count - 1, 1), on_trackbar)
    print("Viewer: a/d or arrow keys = previous/next frame, space = play/pause, slider = jump, q/Esc = quit")

    next_read_index = 0
    shown_index = -1
    try:
        while True:
            if state["index"] != shown_index:
                if state["index"] != next_read_index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, state["index"])
                ok, frame = capture.read()
                if not ok:
                    state["playing"] = False
                    state["index"] = max(0, shown_index)
                    continue
                next_read_index = state["index"] + 1
                shown_index = state["index"]
                cv2.imshow(VIEWER_WINDOW, frame)
                state["suppress"] = True
                cv2.setTrackbarPos("Frame", VIEWER_WINDOW, shown_index)
                state["suppress"] = False

            key = cv2.waitKeyEx(max(1, int(1000 / fps)) if state["playing"] else 30)
            if cv2.getWindowProperty(VIEWER_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state["playing"] = not state["playing"]
            elif key in (ord("d"), 2555904):
                state["playing"] = False
                state["index"] = min(frame_count - 1, state["index"] + 1)
            elif key in (ord("a"), 2424832):
                state["playing"] = False
                state["index"] = max(0, state["index"] - 1)
            elif state["playing"]:
                if state["index"] >= frame_count - 1:
                    state["playing"] = False
                else:
                    state["index"] += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect and track the ball, then review the result frame by frame."
    )
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.15,
        help="Minimum confidence for ball candidates; the tracker filters false positives.",
    )
    parser.add_argument(
        "--max-interpolation-gap",
        type=int,
        default=22,
        help="Fill gaps of at most this many frames between tracked ball positions.",
    )
    parser.add_argument("--device", default="cpu", help="'cpu' or a CUDA device such as '0'.")
    parser.add_argument("--no-viewer", action="store_true", help="Do not open the viewer after processing.")
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Open the viewer on an existing --output-path video without processing.",
    )
    args = parser.parse_args()

    if not args.view_only:
        if args.video_path is None:
            parser.error("--video-path is required unless --view-only is used")
        process_detections(
            video_path=args.video_path,
            output_path=args.output_path,
            ball_conf=args.ball_conf,
            max_interpolation_gap=args.max_interpolation_gap,
            device=args.device,
        )
    if args.view_only or not args.no_viewer:
        view_video(args.output_path)


if __name__ == "__main__":
    main()