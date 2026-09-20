"""
Open the interactive 4-panel review window directly from already-processed
output videos, without re-running detection/tracking/classification/
pitch-projection.

Usage:
    python view_existing_outputs.py
    python view_existing_outputs.py --output-dir outputs
"""
import argparse
from pathlib import Path

from interactive_viewer import MultiViewPlayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    paths = {
        "original": args.output_dir / "original.mp4",
        "annotated": args.output_dir / "annotated.mp4",
        "keypoints": args.output_dir / "keypoints.mp4",
        "projection": args.output_dir / "projection.mp4",
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing output video(s) in {args.output_dir}: {missing}. "
            "Run the full pipeline first (e.g. main.py --interactive) to generate them."
        )

    MultiViewPlayer(paths).run()


if __name__ == "__main__":
    main()
