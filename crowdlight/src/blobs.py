"""
blobs.py  (Lab topics d and e: contour analysis + motion detection)
Finds moving blobs, estimates how many people they hold, and optionally
counts people with OpenCV's HOG detector.
"""
import cv2
import numpy as np


def find_blobs(mask, min_area):
    """Return a list of bounding boxes (x, y, w, h) for contours >= min_area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def find_blobs_with_areas(mask, min_area):
    """Return (boxes, areas): bounding boxes and pixel areas of contours >= min_area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, areas = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            boxes.append(cv2.boundingRect(c))
            areas.append(area)
    return boxes, areas


class AreaCalibrator:
    """Learns the typical area of ONE person from the video itself.

    For the first `frames` frames it collects blob areas. Small blobs are mostly
    single people and big ones are merged groups, so the 25th percentile is a
    good guess for one person. Until it finishes, it returns `fallback`.
    """

    def __init__(self, frames, fallback):
        self.frames = frames
        self.value = float(fallback)
        self.samples = []
        self.seen = 0
        self.done = False

    def update(self, areas):
        """Feed this frame's blob areas; return the current person-size estimate."""
        if not self.done:
            self.samples.extend(areas)
            self.seen += 1
            if self.seen >= self.frames and self.samples:
                self.value = float(np.percentile(self.samples, 25))
                self.done = True
        return self.value


def estimate_people(areas, person_area):
    """Estimate the people count: each blob counts as at least 1 person."""
    return sum(max(1, round(a / person_area)) for a in areas)


class PeopleCounter:
    """Optional secondary stat: counts people with the built-in HOG detector.

    HOG is slow, so it only runs every `every_n` frames and reuses the last
    result in between. It struggles with overlapping people, which is a good
    point for the limitations section of the documentation.
    """

    def __init__(self, every_n):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.every_n = every_n
        self.frame_idx = 0
        self.last_boxes = []

    def detect(self, frame):
        """Return a list of (x, y, w, h) boxes for detected people."""
        if self.frame_idx % self.every_n == 0:
            rects, _ = self.hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            self.last_boxes = [tuple(int(v) for v in r) for r in rects]
        self.frame_idx += 1
        return self.last_boxes
