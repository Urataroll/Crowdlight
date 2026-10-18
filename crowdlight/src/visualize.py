"""
visualize.py
All drawing code: zone overlays, bounding boxes, signal icon, banner and HUD.
"""
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_zones(frame, zones, statuses):
    """Tint each zone with its signal color and label it with name + density.

    Returns the new frame (the tint is blended, so the result is a new image).
    """
    overlay = frame.copy()
    for name, (x1, y1, x2, y2) in zones.items():
        _, color, _ = statuses[name]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    for name, (x1, y1, x2, y2) in zones.items():
        label, color, density = statuses[name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name}: {label} {density * 100:.0f}%"
        cv2.putText(frame, text, (x1 + 10, y1 + 28), FONT, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (x1 + 10, y1 + 28), FONT, 0.7, (255, 255, 255), 2)
    return frame


def draw_blobs(frame, boxes, color=(255, 200, 0)):
    """Draw a bounding box around every detected moving blob."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_people(frame, boxes):
    """Draw HOG people detections (magenta boxes)."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)


def draw_banner(frame, overall, colors):
    """Draw the overall status banner and a small three-light traffic signal."""
    label, color = overall
    h, w = frame.shape[:2]

    # Banner along the bottom
    cv2.rectangle(frame, (0, h - 40), (w, h), color, -1)
    cv2.putText(frame, f"OVERALL CROWD LEVEL: {label}", (15, h - 12), FONT, 0.8, (0, 0, 0), 2)

    # Traffic-signal icon in the top-right corner; only the active light is bright
    cx, top = w - 40, 45
    cv2.rectangle(frame, (cx - 25, top - 35), (cx + 25, top + 95), (30, 30, 30), -1)
    for i, name in enumerate(["RED", "YELLOW", "GREEN"]):
        c = colors[name] if name == label else (70, 70, 70)
        cv2.circle(frame, (cx, top + i * 38), 14, c, -1)


def draw_hud(frame, fps, people_estimate, people_count=None, person_area=None):
    """Draw FPS and counters on the left side (below the zone labels)."""
    shown = "calibrating..." if people_estimate is None else people_estimate
    lines = [f"FPS: {fps:.1f}", f"People (est.): {shown}"]
    if person_area is not None:
        lines.append(f"1 person = {person_area:.0f} px")
    if people_count is not None:
        lines.append(f"HOG people: {people_count}")
    y = 60
    for line in lines:
        cv2.putText(frame, line, (10, y), FONT, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, line, (10, y), FONT, 0.6, (0, 255, 255), 2)
        y += 24


def draw_notice(frame, text):
    """Draw a centered status message (for example while the background is re-learned)."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    x, y = (w - tw) // 2, h // 2
    cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, 0.9, (255, 255, 255), 2)
