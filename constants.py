import supervision as sv

# Must match the class order from data.yaml used during training
CLASS_NAMES = [
    "ball", 
    "goalkeeper", 
    "player", 
    "referee"
]

DEFAULT_CLASS_CONF_THRESHOLDS = {
    0: 0.15,   # ball - recall was the weak point (0.556), so keep this low
    1: 0.35,   # goalkeeper
    2: 0.35,   # player
    3: 0.45,   # referee - raise this to cut false positives from kit-color confusion
}

# Detection classes:
# matches CLASS_NAMES order
CLASS_COLORS = sv.ColorPalette.from_hex([
    "#FFD700",  # ball
    "#00BFFF",  # goalkeeper
    "#FF4136",  # player
    "#B10DC9"   # referee
])

# Display classes:
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

# Must match CLASS_NAMES order: ["ball", "goalkeeper", "player", "referee"]
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

DISPLAY_COLOR_INDEX = {
    BALL_CLASS_ID: 4,
    GOALKEEPER_CLASS_ID: 2,
    REFEREE_CLASS_ID: 3,
}