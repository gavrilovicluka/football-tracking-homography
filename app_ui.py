import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from utils import is_valid_time

from utils import is_valid_time


class ApplicationUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Football Player Tracking")
        self.root.resizable(False, False)

        self.source_type = tk.StringVar(value="file")

        self.youtube_url = tk.StringVar()
        self.video_path = tk.StringVar()

        self.start_time = tk.StringVar(value="00:00:00")
        self.duration = tk.StringVar(value="15")

        self.output_path = tk.StringVar(
            value="outputs/tracked_output.mp4"
        )

        self.confidence = tk.StringVar(value="0.25")
        self.device = tk.StringVar(value="cpu")

        self.result = None

        self._build_ui()
        self._update_source_controls()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

    def _on_close(self):
        self.result = None
        self.root.quit()
        
    def _build_ui(self):
        main_frame = ttk.Frame(
            self.root,
            padding=20,
        )
        main_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        # -------------------------------------------------
        # Input source
        # -------------------------------------------------

        source_frame = ttk.LabelFrame(
            main_frame,
            text="Video source",
            padding=10,
        )
        source_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        ttk.Radiobutton(
            source_frame,
            text="Local MP4 file",
            variable=self.source_type,
            value="file",
            command=self._update_source_controls,
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Radiobutton(
            source_frame,
            text="YouTube URL",
            variable=self.source_type,
            value="youtube",
            command=self._update_source_controls,
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(20, 0),
        )

        # -------------------------------------------------
        # Local video
        # -------------------------------------------------

        ttk.Label(
            source_frame,
            text="Video file:",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        self.video_entry = ttk.Entry(
            source_frame,
            textvariable=self.video_path,
            width=50,
        )
        self.video_entry.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
        )

        self.browse_video_button = ttk.Button(
            source_frame,
            text="Browse...",
            command=self._browse_video,
        )
        self.browse_video_button.grid(
            row=2,
            column=2,
            padx=(10, 0),
        )

        # -------------------------------------------------
        # YouTube
        # -------------------------------------------------

        ttk.Label(
            source_frame,
            text="YouTube URL:",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        self.youtube_entry = ttk.Entry(
            source_frame,
            textvariable=self.youtube_url,
            width=50,
        )
        self.youtube_entry.grid(
            row=4,
            column=0,
            columnspan=3,
            sticky="ew",
        )

        ttk.Label(
            source_frame,
            text="Start time: (HH:MM:SS)",
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(10, 0),
        )

        self.start_entry = ttk.Entry(
            source_frame,
            textvariable=self.start_time,
            width=15,
        )
        self.start_entry.grid(
            row=6,
            column=0,
            sticky="w",
        )

        ttk.Label(
            source_frame,
            text="Duration (seconds):",
        ).grid(
            row=5,
            column=1,
            sticky="w",
            pady=(10, 0),
        )

        self.duration_entry = ttk.Entry(
            source_frame,
            textvariable=self.duration,
            width=15,
        )
        self.duration_entry.grid(
            row=6,
            column=1,
            sticky="w",
        )

        # -------------------------------------------------
        # Processing options
        # -------------------------------------------------

        options_frame = ttk.LabelFrame(
            main_frame,
            text="Processing options",
            padding=10,
        )
        options_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        ttk.Label(
            options_frame,
            text="Confidence:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            options_frame,
            textvariable=self.confidence,
            width=10,
        ).grid(
            row=1,
            column=0,
            sticky="w",
        )

        ttk.Label(
            options_frame,
            text="Device:",
        ).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(20, 0),
        )

        device_combo = ttk.Combobox(
            options_frame,
            textvariable=self.device,
            values=["cpu", "0"],
            state="readonly",
            width=10,
        )
        device_combo.grid(
            row=1,
            column=1,
            sticky="w",
            padx=(20, 0),
        )

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        output_frame = ttk.LabelFrame(
            main_frame,
            text="Output",
            padding=10,
        )
        output_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 10),
        )

        ttk.Label(
            output_frame,
            text="Output video:",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Entry(
            output_frame,
            textvariable=self.output_path,
            width=50,
        ).grid(
            row=1,
            column=0,
            sticky="ew",
        )

        ttk.Button(
            output_frame,
            text="Browse...",
            command=self._browse_output,
        ).grid(
            row=1,
            column=1,
            padx=(10, 0),
        )

        # -------------------------------------------------
        # Start button
        # -------------------------------------------------

        ttk.Button(
            main_frame,
            text="Start processing",
            command=self._submit,
        ).grid(
            row=3,
            column=0,
            sticky="e",
        )

    def _browse_video(self):
        file_path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.video_path.set(file_path)

    def _browse_output(self):
        file_path = filedialog.asksaveasfilename(
            title="Select output video",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
            ],
        )

        if file_path:
            self.output_path.set(file_path)

    def _update_source_controls(self):
        is_file = self.source_type.get() == "file"

        file_state = "normal" if is_file else "disabled"
        youtube_state = "disabled" if is_file else "normal"

        self.video_entry.configure(
            state=file_state
        )
        self.browse_video_button.configure(
            state=file_state
        )

        self.youtube_entry.configure(
            state=youtube_state
        )
        self.start_entry.configure(
            state=youtube_state
        )
        self.duration_entry.configure(
            state=youtube_state
        )

    def _submit(self):
        if self.source_type.get() == "file":
            video_value = self.video_path.get().strip()
            if not video_value:
                messagebox.showerror(
                    "Missing video",
                    "Select a local video file.",
                )
                return

            video_path = Path(video_value)

            if not video_path.exists():
                messagebox.showerror(
                    "Invalid video",
                    "The selected video does not exist.",
                )
                return

            if video_path.suffix.lower() != ".mp4":
                messagebox.showerror(
                    "Invalid video",
                    "Please select MP4 video file.",
                )
                return

        else:
            youtube_url = self.youtube_url.get().strip()

            if not youtube_url:
                messagebox.showerror(
                    "Missing URL",
                    "Enter a YouTube URL.",
                )
                return

        # Validate start time
        start_time = self.start_time.get().strip()

        if start_time and not is_valid_time(start_time):
            messagebox.showerror(
                "Invalid start time",
                "Start time must use HH:MM:SS format.",
            )
            return

        # Validate duration
        try:
            duration = int(self.duration.get())
        except ValueError:
            messagebox.showerror(
                "Invalid duration",
                "Duration must be a whole number.",
            )
            return

        if duration <= 0:
            messagebox.showerror(
                "Invalid duration",
                "Duration must be greater than zero.",
            )
            return

        # Validate confidence
        try:
            confidence = float(self.confidence.get())
        except ValueError:
            messagebox.showerror(
                "Invalid confidence",
                "Confidence must be a number.",
            )
            return

        if not 0.0 < confidence <= 1.0:
            messagebox.showerror(
                "Invalid confidence",
                "Confidence must be between 0 and 1.",
            )
            return

        # Validate output
        output_value = self.output_path.get().strip()

        if not output_value:
            messagebox.showerror(
                "Missing output",
                "Select an output video path.",
            )
            return

        output_path = Path(output_value)

        if output_path.suffix.lower() != ".mp4":
            messagebox.showerror(
                "Invalid output",
                "Output file must have the .mp4 extension.",
            )
            return

        self.result = {
            "source_type": self.source_type.get(),
            "youtube_url": self.youtube_url.get().strip(),
            "video_path": (
                Path(self.video_path.get())
                if self.video_path.get().strip()
                else None
            ),
            "start": self.start_time.get().strip() or None,
            "duration": duration,
            "output": output_path,
            "conf": confidence,
            "device": self.device.get(),
        }

        self.root.quit()

    def run(self):
        self.result = None
        self.root.mainloop()
        return self.result

    def show_error(self, title: str, message: str):
        messagebox.showerror(
            title,
            message,
            parent=self.root,
        )


    def show_info(self, title: str, message: str):
        messagebox.showinfo(
            title,
            message,
            parent=self.root,
        )


    def close(self):
        self.root.destroy()