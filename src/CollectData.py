import csv
import cv2 as cv
import os
import mediapipe as mp
from HandTrackerModule import HandTracker

os.makedirs("data", exist_ok=True)
tracker = HandTracker()
video = cv.VideoCapture(0)

# Make the window appear and stay on top to catch keystrokes
cv.namedWindow('Video', cv.WINDOW_NORMAL)
try:
    cv.setWindowProperty('Video', cv.WND_PROP_TOPMOST, 1)
except cv.error:
    pass

csv_path = 'data/data.csv'
has_header = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

# Use simpler header names moving forward
header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']

with open(csv_path, 'a', newline='') as out:
    writer = csv.writer(out)
    if not has_header:
        writer.writerow(header)
        out.flush()

    print("[INFO] Focus the 'Video' window and press:")
    print("  'o'/'O' for OPEN hand")
    print("  'f'/'F' for FIST")
    print("  't'/'T' for THUMBS UP")
    print("  'p'/'P' for PEACE")
    print("  'q'/'Q' to QUIT")

    while True:
        success, img = video.read()
        if not success:
            break
        quit_flag, wrote = tracker.collectHandData(img, writer)
        if wrote:
            out.flush()
        if quit_flag:
            break

video.release()
cv.destroyAllWindows()
