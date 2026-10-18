"""
main.py
CrowdLight: crowd density monitor with traffic-light signaling.

Usage examples:
    python main.py                          # uses VIDEO_PATH from config.py
    python main.py --profile shibuya        # settings for one video (see PROFILES in config.py)
    python main.py --video videos/hallway.mp4 --save
    python main.py --video 0                # webcam
    python main.py --hog                    # also show the HOG people count

Keys while running:
    q = quit    s = save screenshot    h = toggle heatmap window
"""
import argparse
import os
import time

import cv2

import config
from src.blobs import AreaCalibrator, PeopleCounter, estimate_people, find_blobs_with_areas
from src.heatmap import Heatmap
from src.preprocess import preprocess
from src.segmentation import create_kernel, create_subtractor, get_foreground_mask
from src.traffic_light import ThresholdCalibrator, TrafficLight
from src.visualize import (draw_banner, draw_blobs, draw_hud, draw_notice,
                           draw_people, draw_zones)
from src.zones import all_densities, make_grid_zones, mask_coverage


def parse_args():
    """Read command-line options."""
    p = argparse.ArgumentParser(description="CrowdLight: crowd density monitor")
    p.add_argument("--video", default=None,
                   help="video file path, or 0 for the webcam (default: VIDEO_PATH in config.py)")
    p.add_argument("--profile", choices=sorted(config.PROFILES), default=None,
                   help="use the settings of one video profile from config.py")
    p.add_argument("--save", action="store_true",
                   help=f"save the processed video to {config.OUTPUT_PATH}")
    p.add_argument("--hog", action="store_true",
                   help="also count people with the HOG detector (slower)")
    p.add_argument("--no-display", action="store_true",
                   help="run without windows (useful for testing)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after this many frames (0 = whole video)")
    return p.parse_args()


def open_source(source):
    """Open a video file, or the webcam if `source` is a number like '0'."""
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()
    if args.profile:
        config.apply_profile(args.profile)       # profile settings override the defaults
    source = args.video if args.video is not None else config.VIDEO_PATH
    cap = open_source(source)
    show = not args.no_display

    # ---- Set up all the components ----
    zones = (make_grid_zones(config.FRAME_SIZE, *config.ZONE_GRID)
             if config.AUTO_ZONES else config.ZONES)
    subtractor = create_subtractor(config.MOG2_HISTORY, config.MOG2_VAR_THRESHOLD,
                                   config.DETECT_SHADOWS)
    kernel = create_kernel(config.MORPH_KERNEL_SIZE)
    light = TrafficLight(config.YELLOW_THRESHOLD, config.RED_THRESHOLD,
                         config.SMOOTHING_FRAMES, config.COLORS)
    width, height = config.FRAME_SIZE
    heat = Heatmap((height, width))
    counter = PeopleCounter(config.HOG_EVERY_N_FRAMES) if args.hog else None

    # Auto-calibration: learns the size of one person and the video's normal density
    def new_area_calibrator():
        return AreaCalibrator(config.CALIBRATION_FRAMES, config.PERSON_AREA)

    def new_threshold_calibrator():
        if not config.AUTO_THRESHOLDS:
            return None
        return ThresholdCalibrator(config.CALIBRATION_FRAMES, config.YELLOW_FACTOR,
                                   config.RED_FACTOR, config.YELLOW_FLOOR, config.RED_FLOOR)

    area_calib = new_area_calibrator()
    thr_calib = new_threshold_calibrator()

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(config.OUTPUT_PATH) or ".", exist_ok=True)
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, config.FRAME_SIZE)

    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    show_heat = False
    frame_no = 0
    restart_frame = 0          # frame where the background was last (re)started
    moved_frames = 0           # consecutive frames that look like camera movement
    fps = 0.0
    overall = ("GREEN", config.COLORS["GREEN"])

    # ---- Main loop: one iteration per video frame ----
    while True:
        t0 = time.time()
        ok, raw = cap.read()
        if not ok:
            break
        frame_no += 1

        # 1. Image operations: resize + blur
        frame, blurred = preprocess(raw, config.FRAME_SIZE, config.BLUR_KERNEL)

        # 2. Segmentation: moving foreground mask
        mask = get_foreground_mask(subtractor, blurred, kernel, config.MOG2_LEARNING_RATE)

        # Let MOG2 learn the background before trusting its output
        warming_up = frame_no - restart_frame <= config.WARMUP_FRAMES

        # Camera-movement guard: if most of the frame suddenly "moves" for several
        # frames in a row, the camera moved or the scene cut. Re-learn the background.
        if not warming_up and mask_coverage(mask) > config.CAMERA_MOVE_COVERAGE:
            moved_frames += 1
        else:
            moved_frames = 0
        if moved_frames >= config.CAMERA_MOVE_FRAMES:
            subtractor = create_subtractor(config.MOG2_HISTORY, config.MOG2_VAR_THRESHOLD,
                                           config.DETECT_SHADOWS)
            area_calib = new_area_calibrator()
            thr_calib = new_threshold_calibrator()
            restart_frame = frame_no
            moved_frames = 0
            warming_up = True
            light.history.clear()          # forget the garbage readings from the camera move

        # 3. Analysis: density per zone and blob boxes
        densities = all_densities(mask, zones)
        if warming_up:
            densities = {name: 0.0 for name in densities}
            boxes, areas = [], []
        else:
            heat.update(mask)
            boxes, areas = find_blobs_with_areas(mask, config.MIN_BLOB_AREA)

        # 3b. Auto-calibration (only on frames we trust, i.e. after warm-up)
        person_area = area_calib.value if warming_up else area_calib.update(areas)
        if thr_calib is not None and not warming_up:
            new_thresholds = thr_calib.update(densities)
            if new_thresholds:
                light.set_thresholds(*new_thresholds)
        # Show the estimate only once the person size has been learned
        est_people = estimate_people(areas, person_area) if area_calib.done else None

        # 4. Traffic-light signal per zone, then overall
        statuses = {n: light.update(n, d) for n, d in densities.items()}
        overall = light.overall(statuses)

        # 5. Visualization
        people = counter.detect(frame) if counter else None
        out = draw_zones(frame, zones, statuses)
        draw_blobs(out, boxes)
        if people is not None:
            draw_people(out, people)
        draw_banner(out, overall, config.COLORS)
        draw_hud(out, fps, est_people, len(people) if people is not None else None, person_area)
        if warming_up:
            draw_notice(out, "Camera moved - re-learning background" if restart_frame > 0
                        else "Learning background...")

        if writer:
            writer.write(out)

        if show:
            cv2.imshow("CrowdLight", out)
            cv2.imshow("Foreground mask", mask)
            if show_heat:
                cv2.imshow("Heatmap", heat.render(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("h"):
                show_heat = not show_heat
                if not show_heat:
                    cv2.destroyWindow("Heatmap")
            if key == ord("s"):
                cv2.imwrite(f"{config.SCREENSHOT_DIR}/frame{frame_no}_{overall[0]}.png", out)
                cv2.imwrite(f"{config.SCREENSHOT_DIR}/mask{frame_no}.png", mask)
                cv2.imwrite(f"{config.SCREENSHOT_DIR}/heatmap{frame_no}.png", heat.render(frame))
                print(f"Saved screenshots for frame {frame_no}")

        if args.max_frames and frame_no >= args.max_frames:
            break
        fps = 1.0 / max(time.time() - t0, 1e-6)

    # ---- Clean up ----
    cap.release()
    if writer:
        writer.release()
        print(f"Saved processed video to {config.OUTPUT_PATH}")
    cv2.destroyAllWindows()
    print(f"Processed {frame_no} frames. Final overall level: {overall[0]}")


if __name__ == "__main__":
    main()
