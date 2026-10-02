import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from football_tracking.schema import BALL_CLASS_ID, CLASS_NAMES, GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID
from football_tracking.tracking.ball import BallTrackPoint

# Display palette indices: 0/1 = teams, then goalkeeper, referee, ball, unclassified player
DISPLAY_COLORS = sv.ColorPalette.from_hex([
    "#FF4136",  # team0
    "#0074D9",  # team1
    "#FFDC00",  # goalkeeper
    "#B10DC9",  # referee
    "#00FFFF",  # ball
    "#AAAAAA",  # player without team
])

TEAM_COLORS = {
    0: sv.Color.from_hex("#B71C13"),
    1: sv.Color.from_hex("#4AA8FB"),
}

BALL_COLOR = sv.Color.from_hex("#A2FF00")

DISPLAY_COLOR_INDEX = {
    BALL_CLASS_ID: 4,
    GOALKEEPER_CLASS_ID: 2,
    REFEREE_CLASS_ID: 3,
    PLAYER_CLASS_ID: 5,
}

TILE_ORDER = ["original", "annotated", "keypoints", "projection"]
TILE_TITLES = {
    "original": "Original",
    "annotated": "Player Detections",
    "keypoints": "Pitch Keypoints",
    "projection": "2D Projection",
}
LABEL_HEIGHT = 24
BALL_DETECTED_BGR = (0, 220, 0)
BALL_INTERPOLATED_BGR = (0, 165, 255)


def get_display_color_index(class_id: int, team_id: int) -> int:
    """Maps a detection to a DISPLAY_COLORS index; team_id is -1 when unknown."""
    if class_id == PLAYER_CLASS_ID and team_id != -1:
        return int(team_id)
    return DISPLAY_COLOR_INDEX[class_id]


def build_annotators() -> tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    box_annotator = sv.BoxAnnotator(color=DISPLAY_COLORS, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=DISPLAY_COLORS, text_scale=0.5, text_thickness=1
    )
    return box_annotator, label_annotator


def make_labels(detections: sv.Detections, team_ids: np.ndarray) -> list[str]:
    tracker_ids = detections.tracker_id
    if tracker_ids is None:
        tracker_ids = np.full(len(detections), -1, dtype=int)

    labels = []
    for class_id, tracker_id, team_id in zip(
        detections.class_id, tracker_ids, team_ids
    ):
        name = CLASS_NAMES[class_id]
        tid = f"#{tracker_id}" if tracker_id is not None else ""
        team_tag = f" T{team_id}" if (class_id == PLAYER_CLASS_ID and team_id != -1) else ""
        labels.append(f"{name}{team_tag} {tid}")
    return labels


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    team_ids: np.ndarray,
    annotators: tuple[sv.BoxAnnotator, sv.LabelAnnotator],
    ball_track_point: BallTrackPoint | None = None,
    tracked_ball: bool = False,
) -> np.ndarray:
    """
        Builds labels, picks team/class colors, and draws boxes and labels in one call.
    """
    if tracked_ball:
        non_ball_mask = detections.class_id != BALL_CLASS_ID
        detections = detections[non_ball_mask]
        team_ids = team_ids[non_ball_mask]

    box_annotator, label_annotator = annotators
    labels = make_labels(detections, team_ids)

    # Annotators color by class_id, so substitute the team/display color index
    color_indices = np.array(
        [get_display_color_index(cid, tid) for cid, tid in zip(detections.class_id, team_ids)],
        dtype=int,
    )
    display_detections = sv.Detections(
        xyxy=detections.xyxy.copy(),
        mask=detections.mask.copy() if detections.mask is not None else None,
        confidence=detections.confidence.copy() if detections.confidence is not None else None,
        class_id=color_indices,
        tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
    )

    annotated = box_annotator.annotate(scene=frame.copy(), detections=display_detections)
    annotated = label_annotator.annotate(scene=annotated, detections=display_detections, labels=labels)
    if tracked_ball:
        return draw_ball_track_point(annotated, ball_track_point)
    return annotated


def draw_ball_track_point(
    frame: np.ndarray, point: BallTrackPoint | None
) -> np.ndarray:
    if point is None:
        return frame
    color = BALL_INTERPOLATED_BGR if point.interpolated else BALL_DETECTED_BGR
    center_x, center_y = point.center.astype(int)
    cv2.circle(frame, (center_x, center_y), 7, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y), 9, (0, 0, 0), 1, cv2.LINE_AA)
    label = "ball (estimated)" if point.interpolated else f"ball {point.confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (center_x + 9, center_y - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_keypoints(frame: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """
        Draws circles at the given landmark coordinates on the frame.
    """
    output = frame.copy()
    for x, y in landmark_xy:
        cv2.circle(output, (int(x), int(y)), 5, (0, 0, 255), -1)
    return output


def draw_referee_markers(pitch: np.ndarray, xy: np.ndarray) -> np.ndarray:
    radius = 14
    stripe_width = 4
    scale = 0.1
    padding = 50

    for point in xy:
        center_x = int(point[0] * scale) + padding
        center_y = int(point[1] * scale) + padding
        cv2.circle(pitch, (center_x, center_y), radius, sv.Color.WHITE.as_bgr(), -1)

        marker_size = radius * 2 + 1
        marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
        circle_mask = np.zeros_like(marker)
        cv2.circle(
            circle_mask,
            (radius, radius),
            radius,
            255,
            -1,
        )
        for stripe_start in range(-radius, radius + 1, stripe_width * 2):
            left = stripe_start + radius
            right = min(left + stripe_width, marker_size)
            marker[:, max(left, 0):right] = 255
        marker &= circle_mask

        pitch_height, pitch_width = pitch.shape[:2]
        top = max(center_y - radius, 0)
        bottom = min(center_y + radius + 1, pitch_height)
        left = max(center_x - radius, 0)
        right = min(center_x + radius + 1, pitch_width)
        if top >= bottom or left >= right:
            continue
        mask_top = top - (center_y - radius)
        mask_bottom = mask_top + (bottom - top)
        mask_left = left - (center_x - radius)
        mask_right = mask_left + (right - left)
        marker_crop = marker[mask_top:mask_bottom, mask_left:mask_right]
        pitch[top:bottom, left:right][marker_crop > 0] = sv.Color.BLACK.as_bgr()
        cv2.circle(pitch, (center_x, center_y), radius, sv.Color.BLACK.as_bgr(), 1)

    return pitch


def draw_projection(
    config: SoccerPitchConfiguration,
    pitch_xy: np.ndarray | None,
    team_ids: np.ndarray | None,
    class_ids: np.ndarray | None = None,
    ball_xy: np.ndarray | None = None,
) -> np.ndarray:
    """
        Draws a top-down pitch with players colored by team; pitch_xy is None when there is no homography.
    """
    pitch = draw_pitch(config)

    if pitch_xy is not None and team_ids is not None and len(pitch_xy) > 0:
        if class_ids is None:
            class_ids = np.full(len(pitch_xy), PLAYER_CLASS_ID, dtype=int)

        # Players
        for team_id, color in TEAM_COLORS.items():
            team_mask = (class_ids == PLAYER_CLASS_ID) & (team_ids == team_id)
            if not team_mask.any():
                continue
            pitch = draw_points_on_pitch(
                config=config,
                xy=pitch_xy[team_mask],
                face_color=color,
                edge_color=sv.Color.BLACK,
                radius=14,
                pitch=pitch,
            )

        # Referees
        referee_mask = class_ids == REFEREE_CLASS_ID
        if referee_mask.any():
            pitch = draw_referee_markers(
                pitch,
                pitch_xy[referee_mask],
            )

    # Ball is independent of the detection arrays.
    if ball_xy is not None:
        pitch = draw_points_on_pitch(
            config=config,
            xy=np.asarray(ball_xy, dtype=np.float32).reshape(1, 2),
            face_color=BALL_COLOR,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch,
        )
    return pitch


def label_tile(frame: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((LABEL_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([bar, frame])


def compose_grid(frames: dict[str, np.ndarray], tile_width: int, tile_height: int) -> np.ndarray:
    """
        Builds the labeled 2x2 grid in TILE_ORDER; output is (tile_height + LABEL_HEIGHT) * 2 tall.
    """
    tiles = [
        label_tile(cv2.resize(frames[name], (tile_width, tile_height)), TILE_TITLES[name])
        for name in TILE_ORDER
    ]
    top = cv2.hconcat(tiles[:2])
    bottom = cv2.hconcat(tiles[2:])
    return cv2.vconcat([top, bottom])
