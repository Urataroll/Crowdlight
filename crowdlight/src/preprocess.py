"""
preprocess.py  (Lab topic a: Image Operations)
Resize and blur every frame so the later steps get consistent, less noisy input.
"""
import cv2


def preprocess(frame, size, blur_kernel):
    """Resize a frame to `size` (width, height) and apply a Gaussian blur.

    Returns:
        (resized, blurred): the resized color frame (used for drawing) and the
        blurred copy (used for segmentation).
    """
    resized = cv2.resize(frame, size)
    blurred = cv2.GaussianBlur(resized, blur_kernel, 0)
    return resized, blurred
