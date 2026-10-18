"""
heatmap.py  (Lab topic d: Image Analysis and Transformation)
Accumulates the foreground mask over time to show where people spend the most time.
"""
import cv2
import numpy as np


class Heatmap:
    def __init__(self, shape):
        """`shape` is (height, width) of the frames."""
        self.heat = np.zeros(shape, dtype=np.float32)

    def update(self, mask):
        """Add the current mask (0/255) to the accumulated heat."""
        self.heat += mask.astype(np.float32) / 255.0

    def render(self, background=None):
        """Return a color-mapped heatmap, optionally blended over `background`."""
        peak = self.heat.max()
        if peak <= 0:
            norm = np.zeros(self.heat.shape, dtype=np.uint8)
        else:
            norm = (self.heat / peak * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if background is not None:
            return cv2.addWeighted(background, 0.4, colored, 0.6, 0)
        return colored
