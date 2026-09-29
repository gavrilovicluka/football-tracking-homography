from pathlib import Path

from tqdm import tqdm

from football_tracking.pipeline.analyzer import create_analyzer
from football_tracking.rendering import annotate_frame, build_annotators
from football_tracking.media.video import read_frames, get_video_info, VideoWriter


def process_video(video_path: Path, output_path: Path, conf: float = 0.25, device: str = "cpu"):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    analyzer = create_analyzer(video_path, fps=info["fps"], conf=conf, device=device)
    annotators = build_annotators()
    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])

    unique_ids = set()

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Processing")
        ):
            result = analyzer.process(frame, frame_idx)

            if result.detections.tracker_id is not None:
                unique_ids.update(result.detections.tracker_id.tolist())

            with analyzer.timer.measure("Annotation/write"):
                writer.write(annotate_frame(frame, result.detections, result.team_ids, annotators))
    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")
    print(
        "(Rough sanity check, not a formal metric: expect somewhere around "
        "22 players + ref(s) + ball if tracking stays stable. A much higher "
        "count usually means frequent ID switches from occlusions/re-entries.)"
    )
    analyzer.timer.report()
