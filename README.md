# CrowdLight: Crowd Density Monitor with Traffic-Light Signaling

CSC 126 Computer Vision, Midterm Project (Caraga State University, CCIS)

**Group members:** Elebeth M. Rollorata, Angel Pola Orajay, Jasper Rey Apad Yaun

## Description
CrowdLight watches a fixed-camera video of a school area (hallway, entrance,
canteen line) and measures how crowded each zone is. Each zone is shown as a
**traffic light**: GREEN (free), YELLOW (busy) or RED (congested), plus an
overall status for the whole scene. It uses only classical OpenCV techniques,
with no training and no dataset.

## Lab topics covered
| Topic | Where | What |
|---|---|---|
| a. Image Operations | `src/preprocess.py`, `src/visualize.py` | resize, Gaussian blur, color overlays |
| b. Image Segmentation | `src/segmentation.py` | MOG2 background subtraction, thresholding, morphology |
| d. Image Analysis & Transformation | `src/zones.py`, `src/blobs.py`, `src/heatmap.py` | zone density, contours, heatmap with colormap |
| e. Motion & Object Tracking | `src/segmentation.py`, `src/blobs.py` | motion detection and moving-blob bounding boxes (individual people are not tracked from frame to frame) |
| (extra) HOG people detector | `src/blobs.py` | optional secondary people count |

## How it works
All steps run on every frame inside `src/pipeline.py`, which is shared by the desktop app and the command-line version.

1. **Preprocess:** resize to 960x540 and apply a light Gaussian blur (3x3).
2. **Segment:** MOG2 background subtraction gives a mask of moving people. Shadows are removed by thresholding, and morphology (open/close) cleans the mask. A slow learning rate (`MOG2_LEARNING_RATE`) keeps people who stand still from fading into the background too quickly.
3. **Warm-up:** the first 30 frames are used to learn the background ("Learning background..." is shown). No density is measured yet.
4. **Zones and density:** the frame is split into a 2x2 grid by default (or your own zones). Each zone's **density** = foreground pixels / zone area.
5. **Auto-calibration:** over the next 60 frames the system learns from the video itself:
   - **Size of one person** (25th percentile of blob areas), used for the **People (est.)** counter.
   - **Traffic-light thresholds:** the median density is the scene's "normal" level. YELLOW starts at 1.5x normal and RED at 2.5x normal, with minimums of 8% and 20% so a quiet scene stays green.
6. **Signal:** densities are averaged over the last 15 frames (no flicker) and compared with the thresholds. The overall status is the worst zone.
7. **Camera-move guard:** if more than 50% of the frame is "moving" for 5 frames in a row, the camera has moved or the scene has cut. The background and calibration are reset and re-learned.
8. **Visualize:** zones are tinted and labeled, moving blobs get bounding boxes, and an overall banner and traffic-signal icon are drawn. A heatmap accumulates where people spend the most time.

> The automatic YELLOW/RED levels are **relative to each video**, so they describe "busier than usual for this scene", not an absolute safety limit. To use fixed levels, set `AUTO_THRESHOLDS = False` in `config.py` (defaults: YELLOW at 15%, RED at 35%).

## Setup
```bash
git clone <your-repo-url>
cd crowdlight
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Dependencies: `opencv-python`, `numpy`, `Pillow`. Tkinter comes with Python.

## Run

### Desktop app (recommended)
```bash
python app.py
```
Click **Open video...** (or **Webcam**) and the app starts processing.
- **View:** switch between the annotated result, the foreground mask, and the heatmap
- **Tuning (live):** sliders for the YELLOW/RED thresholds, noise filter and minimum blob size
- **Options:** HOG people count, record the output video
- **Live status:** traffic-light icon plus the status of each zone
- **Playback:** Pause/Resume/Play, Stop, and Screenshot (saves frame, mask and heatmap)

> Linux users: Tkinter may need `sudo apt install python3-tk`. It is included with Python on Windows and macOS.

#### Build a Windows `.exe`
Run this on Windows:
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name CrowdLight app.py
```
The file appears in `dist/CrowdLight.exe`. It is large (100-200 MB) and starts slowly the first time. Test it on the computer you will present on, since antivirus programs sometimes flag PyInstaller files.

### Command line
```bash
python make_sample_video.py                  # optional: create a synthetic test video
python main.py                               # runs on videos/sample.mp4
python main.py --video videos/hallway.mp4 --save   # your own video, saves outputs/result.mp4
python main.py --video 0                     # webcam
python main.py --profile shibuya             # use the settings of one video profile
python main.py --hog                         # add HOG people count
```
**Keys:** `q` quit, `s` save screenshots (frame, mask, heatmap), `h` toggle heatmap window.

### Define your own zones
By default the frame is split into an even grid (`AUTO_ZONES = True`, `ZONE_GRID = (2, 2)`). To draw your own:
```bash
python select_zones.py --video videos/hallway.mp4
```
Drag a rectangle, press ENTER, type a name, repeat. Press ESC when done. Paste the printed `ZONES = {...}` into `config.py` and set `AUTO_ZONES = False`.

### Tuning (all in `config.py`)
- **Too much noise:** raise `MOG2_VAR_THRESHOLD`, `MORPH_KERNEL_SIZE` or `MIN_BLOB_AREA`.
- **Ghosts (people who left still show up):** raise `MOG2_LEARNING_RATE`. **Standing people fade too fast:** lower it.
- **Signal levels:** they are automatic. To adjust them, change `YELLOW_FACTOR`, `RED_FACTOR` or the floors, or set `AUTO_THRESHOLDS = False` and edit `YELLOW_THRESHOLD` / `RED_THRESHOLD`.
- **Signal flickers:** raise `SMOOTHING_FRAMES`.
- **Per video:** add an entry to `PROFILES` at the bottom of `config.py` to keep settings for one video, then run `python main.py --profile NAME`. The included profiles refer to videos (such as `videos/Shibuya.mp4`) that are not in the repository, so add your own footage or edit the paths.

## Project structure
```
crowdlight/
├── app.py               desktop application (Tkinter)
├── main.py              command-line version (OpenCV windows)
├── config.py            all tunable settings and per-video profiles
├── select_zones.py      helper to draw zones
├── make_sample_video.py synthetic test video generator
├── src/                 pipeline (runs all steps on one frame), preprocess,
│                        segmentation, zones, traffic_light (signal and
│                        threshold calibration), blobs (contours, person-size
│                        calibration, HOG), heatmap, visualize
├── videos/              sample input
├── outputs/             processed video
├── screenshots/         sample output screenshots
└── requirements.txt
```

## Limitations
- Needs a **static camera**. If the camera moves, the system detects it and re-learns the background, but readings are unavailable for a moment.
- Density is an **occupancy estimate**, not an exact head count. Overlapping people merge into one blob, so **People (est.)** is only approximate.
- People who stand still for a long time can fade into the background model.
- Lighting changes and reflections can cause false foreground.
- The automatic thresholds are **relative to the video**. They are not absolute crowd limits, and a scene that is always crowded will treat its usual level as normal.
- Perspective: people far from the camera appear smaller and are under-counted.
- Individual people are not tracked from frame to frame.
- Accuracy was checked by observation, not against hand-counted ground truth.

## Possible improvements
Perspective correction per zone, a deep-learning person detector (e.g. YOLO) for exact counts, centroid tracking for flow direction, and logging density over time to a CSV.
