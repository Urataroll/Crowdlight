"""
traffic_light.py
Turns zone densities into GREEN / YELLOW / RED signals.
Densities are averaged over recent frames so the signal does not flicker.
"""
from collections import defaultdict, deque
from statistics import median

# Higher number = more severe. Used to find the "worst" zone.
SEVERITY = {"GREEN": 0, "YELLOW": 1, "RED": 2}


class TrafficLight:
    def __init__(self, yellow_thr, red_thr, window, colors):
        """Keep a rolling window of recent densities for each zone."""
        self.yellow_thr = yellow_thr
        self.red_thr = red_thr
        self.colors = colors
        self.history = defaultdict(lambda: deque(maxlen=window))

    def set_thresholds(self, yellow_thr, red_thr):
        """Replace the thresholds (used by the auto-calibration)."""
        self.yellow_thr = yellow_thr
        self.red_thr = red_thr

    def classify(self, density):
        """Map a density value to a label ('GREEN', 'YELLOW' or 'RED')."""
        if density >= self.red_thr:
            return "RED"
        if density >= self.yellow_thr:
            return "YELLOW"
        return "GREEN"

    def update(self, zone_name, density):
        """Add a new reading and return (label, bgr_color, smoothed_density)."""
        self.history[zone_name].append(density)
        smoothed = sum(self.history[zone_name]) / len(self.history[zone_name])
        label = self.classify(smoothed)
        return label, self.colors[label], smoothed

    def overall(self, statuses):
        """Return (label, bgr_color) of the most severe zone."""
        worst = max((s[0] for s in statuses.values()), key=lambda l: SEVERITY[l])
        return worst, self.colors[worst]


class ThresholdCalibrator:
    """Sets the YELLOW and RED thresholds from this video's own typical density.

    It watches the average zone density for `frames` frames. The median is the
    "normal" level for the scene. YELLOW means busier than normal (x yellow_factor)
    and RED means much busier than normal (x red_factor). The floors keep a
    nearly empty scene from turning yellow because of a little noise.

    Note: the levels are RELATIVE to the video. Say so in the documentation.
    """

    def __init__(self, frames, yellow_factor, red_factor, yellow_floor, red_floor):
        self.frames = frames
        self.yellow_factor = yellow_factor
        self.red_factor = red_factor
        self.yellow_floor = yellow_floor
        self.red_floor = red_floor
        self.samples = []
        self.done = False

    def update(self, densities):
        """Feed one frame's {zone: density}. Returns (yellow, red) once, when ready."""
        if self.done:
            return None
        self.samples.append(sum(densities.values()) / max(len(densities), 1))
        if len(self.samples) < self.frames:
            return None
        self.done = True
        typical = median(self.samples)
        yellow = min(max(self.yellow_floor, typical * self.yellow_factor), 0.90)
        red = min(max(self.red_floor, typical * self.red_factor, yellow + 0.05), 0.95)
        return yellow, red
