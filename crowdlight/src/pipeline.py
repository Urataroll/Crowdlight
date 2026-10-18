"""
pipeline.py
The CrowdLight processing steps wrapped in one class, so the desktop app can
feed it one frame at a time. It runs the same steps as the loop in main.py:
preprocess -> segment -> zone density -> auto-calibrate -> traffic light -> draw.
"""
import time

import cv2

import config
from src.blobs import AreaCalibrator, estimate_people, find_blobs_with_areas
from src.heatmap import Heatmap
from src.preprocess import preprocess
from src.segmentation import create_kernel, create_subtractor, get_foreground_mask
from src.traffic_light import ThresholdCalibrator, TrafficLight
from src.visualize import draw_banner, draw_blobs, draw_hud, draw_notice, draw_zones
from src.zones import all_densities, make_grid_zones, mask_coverage


class CrowdLightPipeline:
    def __init__(self):
        """Create every component from the settings in config.py."""
        self.zones = (make_grid_zones(config.FRAME_SIZE, *config.ZONE_GRID)
                      if config.AUTO_ZONES else config.ZONES)
        self.subtractor = self._new_subtractor()
        self.kernel = create_kernel(config.MORPH_KERNEL_SIZE)
        self.light = TrafficLight(config.YELLOW_THRESHOLD, config.RED_THRESHOLD,
                                  config.SMOOTHING_FRAMES, config.COLORS)
        width, height = config.FRAME_SIZE
        self.heat = Heatmap((height, width))
        self.area_calib = self._new_area_calibrator()
        self.thr_calib = self._new_threshold_calibrator()

        self.frame_no = 0
        self.restart_frame = 0      # frame where the background was last (re)started
        self.moved_frames = 0       # consecutive frames that look like camera movement
        self.last_frame = None      # latest resized frame (used as the heatmap background)
        self.fps = 0.0
        self._last_time = None
        self.overall = ("GREEN", config.COLORS["GREEN"])

    # ---- helpers that build the components ----
    def _new_subtractor(self):
        return create_subtractor(config.MOG2_HISTORY, config.MOG2_VAR_THRESHOLD,
                                 config.DETECT_SHADOWS)

    def _new_area_calibrator(self):
        return AreaCalibrator(config.CALIBRATION_FRAMES, config.PERSON_AREA)

    def _new_threshold_calibrator(self):
        if not config.AUTO_THRESHOLDS:
            return None
        return ThresholdCalibrator(config.CALIBRATION_FRAMES, config.YELLOW_FACTOR,
                                   config.RED_FACTOR, config.YELLOW_FLOOR, config.RED_FLOOR)

    # ---- main entry point ----
    def process(self, raw):
        """Process one raw video frame. Returns (output_frame, foreground_mask)."""
        now = time.time()
        if self._last_time is not None:
            self.fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now
        self.frame_no += 1

        # 1. Image operations: resize + blur
        frame, blurred = preprocess(raw, config.FRAME_SIZE, config.BLUR_KERNEL)
        self.last_frame = frame

        # 2. Segmentation: moving foreground mask
        mask = get_foreground_mask(self.subtractor, blurred, self.kernel,
                                   config.MOG2_LEARNING_RATE)
        warming_up = self.frame_no - self.restart_frame <= config.WARMUP_FRAMES

        # Camera-movement guard: re-learn the background if most of the frame "moves"
        if not warming_up and mask_coverage(mask) > config.CAMERA_MOVE_COVERAGE:
            self.moved_frames += 1
        else:
            self.moved_frames = 0
        if self.moved_frames >= config.CAMERA_MOVE_FRAMES:
            self.subtractor = self._new_subtractor()
            self.area_calib = self._new_area_calibrator()
            self.thr_calib = self._new_threshold_calibrator()
            self.restart_frame = self.frame_no
            self.moved_frames = 0
            warming_up = True
            self.light.history.clear()

        # 3. Analysis: density per zone and blob boxes
        densities = all_densities(mask, self.zones)
        if warming_up:
            densities = {name: 0.0 for name in densities}
            boxes, areas = [], []
        else:
            self.heat.update(mask)
            boxes, areas = find_blobs_with_areas(mask, config.MIN_BLOB_AREA)

        # 3b. Auto-calibration (only on frames we trust, i.e. after warm-up)
        person_area = (self.area_calib.value if warming_up
                       else self.area_calib.update(areas))
        if self.thr_calib is not None and not warming_up:
            new_thresholds = self.thr_calib.update(densities)
            if new_thresholds:
                self.light.set_thresholds(*new_thresholds)
        est_people = (estimate_people(areas, person_area)
                      if self.area_calib.done else None)

        # 4. Traffic-light signal per zone, then overall
        statuses = {n: self.light.update(n, d) for n, d in densities.items()}
        self.overall = self.light.overall(statuses)

        # 5. Visualization
        out = draw_zones(frame, self.zones, statuses)
        draw_blobs(out, boxes)
        draw_banner(out, self.overall, config.COLORS)
        draw_hud(out, self.fps, est_people, None, person_area)
        if warming_up:
            draw_notice(out, "Camera moved - re-learning background" if self.restart_frame > 0
                        else "Learning background...")
        return out, mask

    def heatmap_view(self):
        """Return the heatmap blended over the latest frame."""
        return self.heat.render(self.last_frame)
