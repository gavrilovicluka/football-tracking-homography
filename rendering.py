import supervision as sv

from constants import BALL_CLASS_ID, GOALKEEPER_CLASS_ID, REFEREE_CLASS_ID

# Display palette indices: 0/1 = teams, then goalkeeper, referee, ball
DISPLAY_COLORS = sv.ColorPalette.from_hex([
    "#FF4136",  # team0
    "#0074D9",  # team1
    "#FFDC00",  # goalkeeper
    "#B10DC9",  # referee
    "#FFFFFF"   # ball
])

TEAM_COLORS = {
    0: sv.Color.from_hex("#B71C13"),
    1: sv.Color.from_hex("#4AA8FB"),
}

DISPLAY_COLOR_INDEX = {
    BALL_CLASS_ID: 4,
    GOALKEEPER_CLASS_ID: 2,
    REFEREE_CLASS_ID: 3,
}

TILE_ORDER = ["original", "annotated", "keypoints", "projection"]
TILE_TITLES = {
    "original": "Original",
    "annotated": "Player Detections",
    "keypoints": "Pitch Keypoints",
    "projection": "2D Projection",
}
LABEL_HEIGHT = 24
