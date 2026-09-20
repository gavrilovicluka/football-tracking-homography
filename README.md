## Visualization / Debugging
Visualize detected pitch landmarks:

```bash
python -m scripts.visualize_pitch_landmarks --video-path downloads/clip.mp4 --num-frames 6 --device 0
```

Visualize player projection onto the 2D pitch:
```bash
python -m scripts.visualize_player_projection --video-path downloads/clip.mp4 --num-frames 1 --device 0
```