## Setup
```bash
pip install -e .
```
For the visualization scripts, install the optional debug dependency too:
```bash
pip install -e ".[debug]"
```
Downloading YouTube clips also requires `ffmpeg` (`winget install ffmpeg`) and Deno (`irm https://deno.land/install.ps1 | iex`) for `yt-dlp`.

## Models
| File | Purpose | Used by |
|------|---------|---------|
| `models/football_players_yolo11s_best.pt` | Detects ball, goalkeeper, player, referee | `football_tracking.config.WEIGHTS_PATH` |
| `models/pitch_landmarks_yolo11n_best.pt` | Detects pitch landmarks for homography | `football_tracking.config.PITCH_KEYPOINT_WEIGHTS_PATH` |
| `models/pitch_keypoints_yolo11s_best.pt` | Alternative pitch keypoint model | Not currently used |

## Run full pipeline with interactive window and save videos in \outputs
```bash
python main.py --video-path downloads/clip.mp4 --interactive
```

## Run interactive window with generated outputs
```bash
python view_existing_outputs.py
```

Installed console commands are also available as `football-track` and `football-view`.

## Visualization / Debugging
Visualize detected pitch landmarks:

```bash
python -m scripts.visualize_pitch_landmarks --video-path downloads/clip.mp4 --num-frames 6 --device 0
```

Visualize player projection onto the 2D pitch:
```bash
python -m scripts.visualize_player_projection --video-path downloads/clip.mp4 --num-frames 1 --device 0
```