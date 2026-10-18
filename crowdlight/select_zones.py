"""
select_zones.py
Helper to draw your own zones on the first frame of a video.

Usage:
    python select_zones.py --video videos/hallway.mp4

For each zone: drag a rectangle, press ENTER/SPACE to confirm it, then type
a name in the terminal. Press ESC (or confirm an empty box) when finished.
The result is printed as a ZONES dictionary you can paste into config.py.
"""
import argparse

import cv2

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=config.VIDEO_PATH)
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.video}")
    frame = cv2.resize(frame, config.FRAME_SIZE)

    zones = {}
    while True:
        view = frame.copy()
        for name, (x1, y1, x2, y2) in zones.items():
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(view, name, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        x, y, w, h = cv2.selectROI("Draw a zone (ESC when done)", view, showCrosshair=False)
        if w == 0 or h == 0:
            break
        name = input("Name for this zone: ").strip() or f"Zone {len(zones) + 1}"
        zones[name] = (int(x), int(y), int(x + w), int(y + h))

    cv2.destroyAllWindows()
    print("\nPaste this into config.py:\n")
    print("ZONES = {")
    for name, z in zones.items():
        print(f'    "{name}": {z},')
    print("}")


if __name__ == "__main__":
    main()
