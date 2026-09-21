"""
Interactive 4-way video review window, built on plain OpenCV highgui
(no extra GUI dependency needed).

Default view is a 2x2 grid:
    [ original  | annotated  ]
    [ keypoints | projection ]

Click any panel other than "original" to switch to a 2-panel detail view
with original on the left and the clicked view on the right. Click the
left (original) panel in detail view, or press 'g', to go back to the grid.

The window is resizable - drag its edges or maximize it and the panels scale
to fill the available space while preserving the original video aspect ratio.

Playback is paced by the wall clock rather than "wait N ms, advance one frame":
each tick computes which frame *should* be showing given real elapsed time,
and jumps forward (dropping frames if needed) to match it. This keeps playback
at real speed even if decoding 4 streams per tick is slower than the source
frame rate - a fixed per-frame delay would just compound into visible slow
motion instead.

Controls:
    space / click ▶⏸ button   play / pause
    a / d                     step one frame back / forward (pauses)
    r                         reset to first frame
    g                         back to grid view
    q / ESC                   quit

Usage:
    from interactive_viewer import MultiViewPlayer

    player = MultiViewPlayer({
        "original": "outputs/original.mp4",
        "annotated": "outputs/annotated.mp4",
        "keypoints": "outputs/keypoints.mp4",
        "projection": "outputs/projection.mp4",
    })
    player.run()
"""

import time
from pathlib import Path

import cv2
import numpy as np

from constants import (
    BUTTON_WIDTH,
    CONTROL_BAR_HEIGHT,
    LABEL_HEIGHT,
    TILE_ORDER,
    TILE_TITLES,
)


class MultiViewPlayer:
    def __init__(
        self,
        video_paths: dict[str, Path],
        tile_width: int = 640,
        window_name: str = "Match Review",
    ):
        missing = [k for k in TILE_ORDER if k not in video_paths]
        if missing:
            raise ValueError(f"Missing video paths for: {missing}")

        self.caps = {
            k: cv2.VideoCapture(str(v))
            for k, v in video_paths.items()
        }

        for name, cap in self.caps.items():
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open video for '{name}': {video_paths[name]}"
                )

        # Only seek (.set) when jumping around; sequential playback just
        # calls .read() forward, which is far cheaper than re-seeking
        # every frame.
        self.last_read_idx = {
            name: -1 for name in self.caps
        }

        self.window_name = window_name

        ref = self.caps["original"]

        self.frame_count = int(
            ref.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.fps = ref.get(cv2.CAP_PROP_FPS) or 25.0

        src_w = int(
            ref.get(cv2.CAP_PROP_FRAME_WIDTH)
        ) or 16

        src_h = int(
            ref.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ) or 9

        self.aspect = src_w / src_h

        self.tile_width = tile_width
        self.tile_height = int(
            tile_width / self.aspect
        )

        self.current_frame = 0
        self.playing = False

        self.mode = "grid"          # "grid" or "detail"
        self.detail_panel = None    # one of PANEL_ORDER[1:] when mode == "detail"

        self._suppress_trackbar_cb = False

        # Actual displayed canvas geometry.
        # Used to translate mouse coordinates when the canvas is
        # proportionally scaled and centered inside the window.
        self._display_scale = 1.0
        self._display_offset_x = 0
        self._display_offset_y = 0
        self._display_width = 0
        self._display_height = 0

        # Wall-clock playback pacing state.
        self._play_wall_start = 0.0
        self._play_start_frame = 0

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL,
        )

        cv2.createTrackbar(
            "Frame",
            self.window_name,
            0,
            max(self.frame_count - 1, 1),
            self._on_trackbar,
        )

        cv2.setMouseCallback(
            self.window_name,
            self._on_mouse,
        )

    # ---------------- playback state ----------------

    def _start_playing(self):
        self.playing = True
        self._play_wall_start = time.perf_counter()
        self._play_start_frame = self.current_frame

    def _pause(self):
        self.playing = False

    def _toggle_play(self):
        if self.playing:
            self._pause()
        else:
            self._start_playing()

    def _advance_by_wall_clock(self):
        if not self.playing:
            return

        elapsed = (
            time.perf_counter() -
            self._play_wall_start
        )

        target = (
            self._play_start_frame +
            int(elapsed * self.fps)
        )

        if target >= self.frame_count - 1:
            self.current_frame = self.frame_count - 1
            self._pause()
        else:
            self.current_frame = target

    # ---------------- event handlers ----------------

    def _on_trackbar(self, value):
        if self._suppress_trackbar_cb:
            return

        self._pause()
        self.current_frame = value

    def _window_to_canvas_coordinates(self, x, y):
        """
        Convert mouse coordinates from the OpenCV window into coordinates
        of the internally generated canvas.

        The canvas may be scaled and centered inside the window, so this
        removes the scaling and black margins first.
        """
        if self._display_scale <= 0:
            return None

        canvas_x = (
            x - self._display_offset_x
        ) / self._display_scale

        canvas_y = (
            y - self._display_offset_y
        ) / self._display_scale

        return canvas_x, canvas_y

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        coordinates = self._window_to_canvas_coordinates(
            x,
            y,
        )

        if coordinates is None:
            return

        canvas_x, canvas_y = coordinates

        # Ignore clicks outside the actual canvas.
        canvas_width = self._display_width / self._display_scale
        canvas_height = self._display_height / self._display_scale

        if (
            canvas_x < 0
            or canvas_y < 0
            or canvas_x >= canvas_width
            or canvas_y >= canvas_height
        ):
            return

        if canvas_y < CONTROL_BAR_HEIGHT:
            if canvas_x < BUTTON_WIDTH:
                self._toggle_play()

            return

        video_y = canvas_y - CONTROL_BAR_HEIGHT

        if self.mode == "grid":
            col = 0 if canvas_x < self.tile_width else 1
            row = 0 if video_y < self.tile_height else 1

            panel = TILE_ORDER[row * 2 + col]

            if panel != "original":
                self.mode = "detail"
                self.detail_panel = panel

        else:
            if canvas_x < self.tile_width:
                self.mode = "grid"
                self.detail_panel = None

    # ---------------- sizing ----------------

    def _resize_keep_aspect(
        self,
        frame,
        target_width,
        target_height,
    ):
        """
        Resize a frame to fit inside target dimensions while preserving
        its original aspect ratio.

        Empty space is filled with black pixels.
        """
        src_height, src_width = frame.shape[:2]

        if src_width <= 0 or src_height <= 0:
            return np.zeros(
                (target_height, target_width, 3),
                dtype=np.uint8,
            )

        scale = min(
            target_width / src_width,
            target_height / src_height,
        )

        new_width = max(
            1,
            int(src_width * scale),
        )

        new_height = max(
            1,
            int(src_height * scale),
        )

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        canvas = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8,
        )

        x = (
            target_width - new_width
        ) // 2

        y = (
            target_height - new_height
        ) // 2

        canvas[
            y:y + new_height,
            x:x + new_width,
        ] = resized

        return canvas

    def _fit_canvas_to_window(self, canvas):
        """
        Scale the complete viewer canvas to fit inside the current window
        while preserving its aspect ratio.

        The remaining area is filled with black pixels.
        """
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(
                self.window_name
            )
        except cv2.error:
            return canvas

        if win_w <= 0 or win_h <= 0:
            return canvas

        canvas_height, canvas_width = canvas.shape[:2]

        scale = min(
            win_w / canvas_width,
            win_h / canvas_height,
        )

        new_width = max(
            1,
            int(canvas_width * scale),
        )

        new_height = max(
            1,
            int(canvas_height * scale),
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        resized = cv2.resize(
            canvas,
            (new_width, new_height),
            interpolation=interpolation,
        )

        output = np.zeros(
            (win_h, win_w, 3),
            dtype=np.uint8,
        )

        offset_x = (
            win_w - new_width
        ) // 2

        offset_y = (
            win_h - new_height
        ) // 2

        output[
            offset_y:offset_y + new_height,
            offset_x:offset_x + new_width,
        ] = resized

        # Store geometry for mouse coordinate conversion.
        self._display_scale = scale
        self._display_offset_x = offset_x
        self._display_offset_y = offset_y
        self._display_width = new_width
        self._display_height = new_height

        return output

    # ---------------- rendering ----------------

    def _read(self, panel: str, frame_idx: int):
        cap = self.caps[panel]

        last = self.last_read_idx[panel]

        if frame_idx != last + 1:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_idx,
            )

        ok, frame = cap.read()

        video_height = (
            self.tile_height -
            LABEL_HEIGHT
        )

        if not ok:
            return np.zeros(
                (
                    self.tile_height,
                    self.tile_width,
                    3,
                ),
                dtype=np.uint8,
            )

        self.last_read_idx[panel] = frame_idx

        return self._resize_keep_aspect(
            frame,
            self.tile_width,
            video_height,
        )

    def _label(self, frame, text):
        bar = np.zeros(
            (
                LABEL_HEIGHT,
                frame.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            bar,
            text,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return cv2.vconcat(
            [bar, frame]
        )

    def _draw_control_bar(self, width: int):
        bar = np.full(
            (
                CONTROL_BAR_HEIGHT,
                width,
                3,
            ),
            40,
            dtype=np.uint8,
        )

        cx = BUTTON_WIDTH // 2
        cy = CONTROL_BAR_HEIGHT // 2

        if self.playing:
            cv2.rectangle(
                bar,
                (cx - 7, cy - 8),
                (cx - 2, cy + 8),
                (255, 255, 255),
                -1,
            )

            cv2.rectangle(
                bar,
                (cx + 2, cy - 8),
                (cx + 7, cy + 8),
                (255, 255, 255),
                -1,
            )

        else:
            triangle = np.array(
                [
                    [cx - 7, cy - 9],
                    [cx - 7, cy + 9],
                    [cx + 9, cy],
                ]
            )

            cv2.fillConvexPoly(
                bar,
                triangle,
                (255, 255, 255),
            )

        text = (
            f"Frame "
            f"{self.current_frame + 1} / "
            f"{self.frame_count}"
        )

        cv2.putText(
            bar,
            text,
            (BUTTON_WIDTH + 12, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return bar

    def _build_canvas(self, frame_idx: int):
        if self.mode == "grid":
            tiles = [
                self._label(
                    self._read(panel, frame_idx),
                    TILE_TITLES[panel],
                )
                for panel in TILE_ORDER
            ]

            top = cv2.hconcat(
                [tiles[0], tiles[1]]
            )

            bottom = cv2.hconcat(
                [tiles[2], tiles[3]]
            )

            video_area = cv2.vconcat(
                [top, bottom]
            )

        else:
            left = self._label(
                self._read(
                    "original",
                    frame_idx,
                ),
                TILE_TITLES["original"],
            )

            right = self._label(
                self._read(
                    self.detail_panel,
                    frame_idx,
                ),
                TILE_TITLES[self.detail_panel],
            )

            video_area = cv2.hconcat(
                [left, right]
            )

        control_bar = self._draw_control_bar(
            video_area.shape[1]
        )

        return cv2.vconcat(
            [control_bar, video_area]
        )

    # ---------------- main loop ----------------

    def run(self):
        while True:
            self._advance_by_wall_clock()

            canvas = self._build_canvas(
                self.current_frame
            )

            # Preserve the complete canvas aspect ratio when the
            # OpenCV window is resized.
            display_canvas = self._fit_canvas_to_window(
                canvas
            )

            cv2.imshow(
                self.window_name,
                display_canvas,
            )

            self._suppress_trackbar_cb = True

            cv2.setTrackbarPos(
                "Frame",
                self.window_name,
                self.current_frame + 1,
            )

            self._suppress_trackbar_cb = False

            # Pacing now comes from the wall clock above, not from this
            # delay - keep it short so key/mouse events stay responsive
            # and we re-check the clock as often as rendering allows.
            key = cv2.waitKey(
                1 if self.playing else 30
            ) & 0xFF

            if key == ord(" "):
                self._toggle_play()

            elif key in (ord("q"), 27):
                break

            elif key == ord("g"):
                self.mode = "grid"
                self.detail_panel = None

            elif key == ord("r"):
                self._pause()
                self.current_frame = 0

            elif key == ord("a"):
                self._pause()
                self.current_frame = max(
                    0,
                    self.current_frame - 1,
                )

            elif key == ord("d"):
                self._pause()
                self.current_frame = min(
                    self.frame_count - 1,
                    self.current_frame + 1,
                )

            if cv2.getWindowProperty(
                self.window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

        self.close()

    def close(self):
        for cap in self.caps.values():
            cap.release()

        cv2.destroyWindow(
            self.window_name
        )