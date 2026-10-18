"""
zones.py  (Lab topic d: Image Analysis)
Measures how crowded each region of the frame is.
"""
import cv2


def zone_density(mask, zone):
    """Fraction (0.0 - 1.0) of pixels inside `zone` that are foreground.

    `zone` is (x1, y1, x2, y2). Coordinates are clamped to the mask size.
    """
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = zone
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    return cv2.countNonZero(mask[y1:y2, x1:x2]) / area


def all_densities(mask, zones):
    """Return {zone_name: density} for every zone."""
    return {name: zone_density(mask, zone) for name, zone in zones.items()}


def make_grid_zones(size, rows, cols):
    """Split a frame of `size` (width, height) into an even rows x cols grid.

    Zones are named Zone A, Zone B, ... left to right, top to bottom.
    """
    w, h = size
    zones = {}
    for r in range(rows):
        for c in range(cols):
            name = f"Zone {chr(ord('A') + r * cols + c)}"
            zones[name] = (c * w // cols, r * h // rows,
                           (c + 1) * w // cols, (r + 1) * h // rows)
    return zones


def mask_coverage(mask):
    """Fraction (0.0 - 1.0) of the whole frame that is foreground."""
    return cv2.countNonZero(mask) / mask.size
