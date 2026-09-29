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

from config import OUTPUTS_DIR, multiview_output_paths
from interactive_viewer import MultiViewPlayer
from rendering import TILE_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    all_paths = multiview_output_paths(args.output_dir)
    paths = {name: all_paths[name] for name in TILE_ORDER}

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing output video(s) in {args.output_dir}: {missing}. "
            "Run the full pipeline first (e.g. main.py --interactive) to generate them."
        )

    MultiViewPlayer(paths).run()


if __name__ == "__main__":
    main()
