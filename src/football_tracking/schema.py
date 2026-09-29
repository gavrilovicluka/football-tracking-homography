# Must match the class order from data.yaml used during training
CLASS_NAMES = [
    "ball", 
    "goalkeeper", 
    "player", 
    "referee"
]

# Must match CLASS_NAMES order: ["ball", "goalkeeper", "player", "referee"]
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

DEFAULT_CLASS_CONF_THRESHOLDS = {
    BALL_CLASS_ID: 0.15,        # recall was the weak point (0.556), so keep this low
    GOALKEEPER_CLASS_ID: 0.35,
    PLAYER_CLASS_ID: 0.35,
    REFEREE_CLASS_ID: 0.45,     # raise this to cut false positives from kit-color confusion
}
