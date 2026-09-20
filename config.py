WEIGHTS_PATH = "models/football_players_yolo11s_best.pt"
PITCH_KEYPOINT_WEIGHTS_PATH = "models/pitch_landmarks_yolo11n_best.pt"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

PLAYER_CROPS_SAMPLE_COUNT = 10  # number of player crops to sample across the clip to fit team classifier

BATCH_SIZE = 32  # batch size for embedding extraction

# IMAGE_SIZE = 1280
IMAGE_SIZE = 960

CLASSIFICATION_INTERVAL = 3