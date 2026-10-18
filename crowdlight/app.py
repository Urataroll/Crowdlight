"""
app.py
CrowdLight desktop application (Tkinter).

Run with:   python app.py

Pick a webcam or a video file, press Start, and watch the crowd level of each
zone as a traffic light. Uses the same processing as main.py (see src/pipeline.py).
"""
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import config
from src.pipeline import CrowdLightPipeline


class CrowdLightApp:
    def __init__(self, root):
        self.root = root
        root.title("CrowdLight - Crowd Density Monitor")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.cap = None
        self.pipeline = None
        self.running = False
        self.after_id = None
        self.frame_delay = 1          # ms between frames (matches the video's speed)
        self.photo = None             # keeps a reference so Tk does not discard the image

        # Values bound to the widgets
        self.source = tk.StringVar(value="video")
        self.camera_index = tk.IntVar(value=0)
        default_video = config.VIDEO_PATH if os.path.exists(config.VIDEO_PATH) else ""
        self.video_path = tk.StringVar(value=default_video)
        self.view = tk.StringVar(value="result")
        self.status = tk.StringVar(value="Choose a source, then press Start.")

        self.build_ui()

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        """Create all the widgets."""
        # --- Source selection ---
        src = tk.LabelFrame(self.root, text="Source", padx=8, pady=6)
        src.pack(fill="x", padx=10, pady=(10, 4))

        tk.Radiobutton(src, text="Webcam, camera number:", variable=self.source,
                       value="webcam").grid(row=0, column=0, sticky="w")
        tk.Spinbox(src, from_=0, to=5, width=3,
                   textvariable=self.camera_index).grid(row=0, column=1, sticky="w")

        tk.Radiobutton(src, text="Video file:", variable=self.source,
                       value="video").grid(row=1, column=0, sticky="w")
        tk.Entry(src, textvariable=self.video_path, width=60).grid(row=1, column=1, padx=4)
        tk.Button(src, text="Browse...", command=self.browse).grid(row=1, column=2)

        # --- Controls ---
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=10, pady=4)
        self.start_btn = tk.Button(bar, text="Start", width=10, command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(bar, text="Stop", width=10, state="disabled",
                                  command=lambda: self.stop("Stopped."))
        self.stop_btn.pack(side="left", padx=6)

        tk.Label(bar, text="View:").pack(side="left", padx=(20, 2))
        for text, value in (("Result", "result"), ("Mask", "mask"), ("Heatmap", "heatmap")):
            tk.Radiobutton(bar, text=text, variable=self.view, value=value).pack(side="left")

        # --- Video panel (starts as a black picture the size of the output) ---
        width, height = config.FRAME_SIZE
        blank = ImageTk.PhotoImage(Image.new("RGB", (width, height), "black"))
        self.panel = tk.Label(self.root, image=blank, bg="black")
        self.panel.image = blank
        self.panel.pack(padx=10, pady=4)

        # --- Status bar ---
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 relief="sunken").pack(fill="x", padx=10, pady=(0, 10))

    def browse(self):
        """Let the user choose a video file."""
        path = filedialog.askopenfilename(
            title="Choose a video",
            initialdir="videos" if os.path.isdir("videos") else ".",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.video_path.set(path)
            self.source.set("video")

    # ------------------------------------------------------- start / stop
    def open_capture(self):
        """Open the chosen source. Returns a VideoCapture, or None if it failed."""
        if self.source.get() == "webcam":
            index = self.camera_index.get()
            # DirectShow avoids a slow start / black window on Windows
            cap = (cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32"
                   else cv2.VideoCapture(index))
            self.frame_delay = 1
        else:
            path = self.video_path.get().strip()
            if not os.path.isfile(path):
                messagebox.showerror("CrowdLight", "Please choose a valid video file.")
                return None
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = int(1000 / fps) if fps and fps > 0 else 33
        if not cap.isOpened():
            messagebox.showerror("CrowdLight", "Could not open the selected source.")
            return None
        return cap

    def start(self):
        """Open the source, create a fresh pipeline and begin processing."""
        if self.running:
            return
        cap = self.open_capture()
        if cap is None:
            return
        self.cap = cap
        self.pipeline = CrowdLightPipeline()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.set("Running...")
        self.tick()

    def stop(self, message):
        """Stop processing and release the video source."""
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.set(message)

    # ---------------------------------------------------------- main loop
    def tick(self):
        """Process one frame, show it, then schedule the next one."""
        if not self.running:
            return
        t0 = time.time()
        ok, raw = self.cap.read()
        if not ok:
            self.stop("Finished: end of video." if self.source.get() == "video"
                      else "The webcam stopped sending frames.")
            return

        out, mask = self.pipeline.process(raw)
        self.show(self.pick_view(out, mask))
        self.status.set(f"Running  |  Overall crowd level: {self.pipeline.overall[0]}")

        elapsed = int((time.time() - t0) * 1000)
        self.after_id = self.root.after(max(1, self.frame_delay - elapsed), self.tick)

    def pick_view(self, out, mask):
        """Return the image for the selected view (result, mask or heatmap)."""
        if self.view.get() == "mask":
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.view.get() == "heatmap":
            return self.pipeline.heatmap_view()
        return out

    def show(self, bgr_image):
        """Display an OpenCV (BGR) image in the video panel."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.panel.config(image=self.photo)

    def on_close(self):
        """Release the camera/video before closing the window."""
        self.stop("")
        self.root.destroy()


def main():
    root = tk.Tk()
    CrowdLightApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
