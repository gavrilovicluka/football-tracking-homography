from pathlib import Path

from tqdm import tqdm

from football_tracking.pipeline.analyzer import create_analyzer, track_ball_results
from football_tracking.pipeline.analysis_cache import create_frame_analysis_cache
from football_tracking.rendering import annotate_frame, build_annotators
from football_tracking.media.video import read_frames, get_video_info, VideoWriter


def process_video(
    video_path: Path,
    output_path: Path,
    conf: float = 0.25,
    ball_conf: float = 0.15,
    device: str = "cpu",
):
    info = get_video_info(video_path)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['frame_count']} frames")

    analysis_cache = create_frame_analysis_cache(
        video_path=video_path,
        frame_count=info["frame_count"],
        fps=info["fps"],
        conf=conf,
        ball_conf=ball_conf,
        device=device,
        with_pitch=False,
        pitch_conf=0.0,
        pitch_imgsz=0,
        pitch_detect_interval=0,
    )
    analyzer = None
    if analysis_cache.hit:
        print("Using cached frame analysis; detection and classification are skipped.")
    else:
        analyzer = create_analyzer(
            video_path,
            fps=info["fps"],
            conf=conf,
            ball_conf=ball_conf,
            device=device,
            with_pitch=False,
        )
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Analyzing")
        ):
            analysis_cache.record(analyzer.process(frame, frame_idx))

    track_ball_results(analysis_cache.results)
    if not analysis_cache.hit:
        analysis_cache.save()

    annotators = build_annotators()
    writer = VideoWriter(output_path, fps=info["fps"], width=info["width"], height=info["height"])

    unique_ids = set()

    try:
        for frame_idx, frame in enumerate(
            tqdm(read_frames(video_path), total=info["frame_count"], desc="Rendering")
        ):
            result = analysis_cache.results[frame_idx]

            if result.detections.tracker_id is not None:
                unique_ids.update(result.detections.tracker_id.tolist())

            if analyzer is not None:
                with analyzer.timer.measure("Annotation/write"):
                    writer.write(annotate_frame(
                        frame,
                        result.detections,
                        result.team_ids,
                        annotators,
                        ball_track_point=result.ball_track_point,
                        tracked_ball=True,
                    ))
            else:
                writer.write(annotate_frame(
                    frame,
                    result.detections,
                    result.team_ids,
                    annotators,
                    ball_track_point=result.ball_track_point,
                    tracked_ball=True,
                ))
    finally:
        writer.release()

    print(f"\nDone. Saved annotated video to: {output_path}")
    print(f"Unique track IDs seen across the clip: {len(unique_ids)}")
    print(
        "(Rough sanity check, not a formal metric: expect somewhere around "
        "22 players + ref(s) + ball if tracking stays stable. A much higher "
        "count usually means frequent ID switches from occlusions/re-entries.)"
    )
    if analyzer is not None:
        analyzer.timer.report()
