"""
config.py
---------
Every tunable value for CrowdLight lives here, so tuning never means
digging through the other files.

Most values are now learned from the video automatically (person size,
traffic-light thresholds, zones). The numbers below are starting points.
Per-video overrides live in PROFILES at the bottom: run with --profile NAME.
"""

# ---------- Input / output ----------
VIDEO_PATH = "videos/sample.mp4"       # default video (or pass --video 0 for webcam)
OUTPUT_PATH = "outputs/result.mp4"      # where the processed video is saved with --save
SCREENSHOT_DIR = "screenshots"          # where the 's' key saves screenshots

# ---------- Preprocessing ----------
FRAME_SIZE = (960, 540)                 # (width, height) every frame is resized to
BLUR_KERNEL = (3, 3)                    # Gaussian blur kernel (must be odd numbers)

# ---------- Segmentation (MOG2 background subtraction) ----------
MOG2_HISTORY = 500                      # frames used to learn the background
MOG2_VAR_THRESHOLD = 25                 # lower = more sensitive, higher = less noise
DETECT_SHADOWS = True                   # shadows are marked gray (127) then removed
MORPH_KERNEL_SIZE = 3                   # size of the elliptical morphology kernel
WARMUP_FRAMES = 30                      # frames to let MOG2 learn before measuring density
MOG2_LEARNING_RATE = 0.004              # higher = forgets people who left faster (fewer ghosts)

# ---------- Zones ----------
AUTO_ZONES = True                       # True = split the frame into an even grid
ZONE_GRID = (2, 2)                      # (rows, columns) used when AUTO_ZONES is True
# Used only when AUTO_ZONES is False: name -> (x1, y1, x2, y2) in FRAME_SIZE pixels.
# Run `python select_zones.py` to draw your own and paste the output here.
ZONES = {
    "Zone A": (0, 0, 480, 270),
    "Zone B": (480, 0, 960, 270),
    "Zone C": (0, 270, 480, 540),
    "Zone D": (480, 270, 960, 540),
}

# ---------- Traffic-light thresholds (fraction of a zone covered by people) ----------
YELLOW_THRESHOLD = 0.15                 # density >= this  -> YELLOW (busy)
RED_THRESHOLD = 0.35                    # density >= this  -> RED (congested)
SMOOTHING_FRAMES = 15                   # average over this many frames to stop flicker

# ---------- Auto-calibration (learned from the video itself) ----------
CALIBRATION_FRAMES = 60                 # frames used (after warm-up) to learn the video's scale
AUTO_THRESHOLDS = True                  # True = set the thresholds from this video's typical density
YELLOW_FACTOR = 1.5                     # YELLOW when density is 1.5x the typical (median) density
RED_FACTOR = 2.5                        # RED when density is 2.5x the typical density
YELLOW_FLOOR = 0.08                     # never go below these, so an empty scene stays GREEN
RED_FLOOR = 0.20

# ---------- Camera-movement guard ----------
CAMERA_MOVE_COVERAGE = 0.5              # if more than 50% of the frame is "moving"
CAMERA_MOVE_FRAMES = 5                  # for this many frames in a row, assume the camera moved

# ---------- Blob detection ----------
MIN_BLOB_AREA = 150                     # ignore contours smaller than this (pixels)
PERSON_AREA = 150                       # fallback size of one person until calibration finishes

# ---------- Optional HOG people counter ----------
HOG_EVERY_N_FRAMES = 5                  # HOG is slow, so run it every N frames only

# ---------- Colors (BGR) ----------
COLORS = {
    "GREEN": (0, 200, 0),
    "YELLOW": (0, 220, 255),
    "RED": (0, 0, 255),
}

# ---------- Per-video profiles: python main.py --profile shibuya ----------
# Each profile overrides any setting above. Add your own videos here.
PROFILES = {
    "shibuya": {
        "VIDEO_PATH": "videos/Shibuya.mp4",
        "MOG2_VAR_THRESHOLD": 25,
        "MIN_BLOB_AREA": 150,
    },
    "crowd_area": {
        "VIDEO_PATH": "videos/crowd_area.mp4",
        "MIN_BLOB_AREA": 400,
    },
    "sample": {
        "VIDEO_PATH": "videos/sample.mp4",
        "AUTO_THRESHOLDS": False,
    },
}


def apply_profile(name):
    """Copy the settings of PROFILES[name] over the defaults above."""
    globals().update(PROFILES[name])
