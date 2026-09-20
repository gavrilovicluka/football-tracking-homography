## Run full pipeline with interactive window and save videos in \outputs
```bash
python main.py --video-path downloads/clip.mp4 --interactive
```

## Run interactive window with generated outputs
```bash
python view_existing_outputs.py
```

## Visualization / Debugging
Visualize detected pitch landmarks:

```bash
python -m scripts.visualize_pitch_landmarks --video-path downloads/clip.mp4 --num-frames 6 --device 0
```

Visualize player projection onto the 2D pitch:
```bash
python -m scripts.visualize_player_projection --video-path downloads/clip.mp4 --num-frames 1 --device 0
```