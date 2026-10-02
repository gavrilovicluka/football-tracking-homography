from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

WEIGHTS_PATH = MODELS_DIR / "football_players_yolo11m_best-1280.pt"
PITCH_KEYPOINT_WEIGHTS_PATH = MODELS_DIR / "pitch_landmarks_yolo11m_best.pt"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

DEFAULT_CLIP_PATH = DOWNLOADS_DIR / "bay-stu.mp4"
DEFAULT_OUTPUT_PATH = OUTPUTS_DIR / "tracked_output.mp4"

MULTIVIEW_FILENAMES = {
    "original": "original.mp4",
    "annotated": "annotated.mp4",
    "keypoints": "keypoints.mp4",
    "projection": "projection.mp4",
    "combined": "combined.mp4",
}


def multiview_output_paths(output_dir: str | Path = OUTPUTS_DIR) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        name: output_dir / filename
        for name, filename in MULTIVIEW_FILENAMES.items()
    }

PLAYER_CROPS_SAMPLE_COUNT = 10  # number of player crops to sample across the clip to fit team classifier

BATCH_SIZE = 32  # batch size for embedding extraction

IMAGE_SIZE = 1280
# IMAGE_SIZE = 960
PITCH_IMAGE_SIZE = IMAGE_SIZE
PITCH_CONFIDENCE = 0.5
PITCH_DETECT_INTERVAL = 1

CLASSIFICATION_INTERVAL = 3

LABEL_HEIGHT = 24

MAX_INTERPOLATION_GAP = 25  # maximum gap for ball track interpolation