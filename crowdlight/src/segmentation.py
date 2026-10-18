"""
segmentation.py  (Lab topic b: Image Segmentation)
Separates moving people (foreground) from the static background.
"""
import cv2


def create_subtractor(history, var_threshold, detect_shadows):
    """Return a configured MOG2 background subtractor."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )


def create_kernel(size):
    """Return an elliptical structuring element for morphology."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def get_foreground_mask(subtractor, blurred_frame, kernel, learning_rate=-1):
    """Return a clean binary mask (white = moving person, black = background).

    Steps:
        1. MOG2 gives 255 for foreground, 127 for shadow, 0 for background.
        2. Thresholding at 200 drops the shadow pixels.
        3. Opening removes small specks of noise.
        4. Closing fills small holes inside people blobs.

    learning_rate: how fast the background adapts (-1 = OpenCV's automatic rate).
    """
    raw = subtractor.apply(blurred_frame, learningRate=learning_rate)
    _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask
