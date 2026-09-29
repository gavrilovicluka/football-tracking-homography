import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from football_tracking.schema import BALL_CLASS_ID, CLASS_NAMES, GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID

# Display palette indices: 0/1 = teams, then goalkeeper, referee, ball, unclassified player
DISPLAY_COLORS = sv.ColorPalette.from_hex([
    "#FF4136",  # team0
    "#0074D9",  # team1
    "#FFDC00",  # goalkeeper
    "#B10DC9",  # referee
    "#FFFFFF",  # ball
    "#AAAAAA",  # player without team
])

TEAM_COLORS = {
    0: sv.Color.from_hex("#B71C13"),
    1: sv.Color.from_hex("#4AA8FB"),
}

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
    labels = []
    for class_id, tracker_id, team_id in zip(
        detections.class_id, detections.tracker_id, team_ids
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
) -> np.ndarray:
    """
        Builds labels, picks team/class colors, and draws boxes and labels in one call.
    """
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
    return label_annotator.annotate(scene=annotated, detections=display_detections, labels=labels)


def draw_keypoints(frame: np.ndarray, landmark_xy: np.ndarray) -> np.ndarray:
    """
        Draws circles at the given landmark coordinates on the frame.
    """
    output = frame.copy()
    for x, y in landmark_xy:
        cv2.circle(output, (int(x), int(y)), 5, (0, 0, 255), -1)
    return output


def draw_projection(
    config: SoccerPitchConfiguration,
    pitch_xy: np.ndarray | None,
    team_ids: np.ndarray | None,
) -> np.ndarray:
    """
        Draws a top-down pitch with players colored by team; pitch_xy is None when there is no homography.
    """
    pitch = draw_pitch(config)
    if pitch_xy is None or team_ids is None or len(pitch_xy) == 0:
        return pitch

    for team_id, color in TEAM_COLORS.items():
        team_mask = team_ids == team_id
        if team_mask.any():
            pitch = draw_points_on_pitch(
                config=config,
                xy=pitch_xy[team_mask],
                face_color=color,
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
