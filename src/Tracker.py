from pathlib import Path
import json
import time
import joblib
import numpy as np
import cv2 as cv

from HandTrackerModule import HandTracker


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"
META_PATH = MODELS_DIR / "random_forest.meta.json"
model = joblib.load(MODEL_PATH)
meta = {}
if META_PATH.exists():
    with open(META_PATH, "r") as f:
        meta = json.load(f)
label_map = meta.get("label_map", {0: "open", 1: "fist", 2: "thumbs_up", 3: "peace"})
feature_count = meta.get("feature_count", 63)
dropZ = (feature_count == 42)
cTime, pTime = 0, 0

tracker = HandTracker(detectionCon=0.7)
video = cv.VideoCapture(0)

cv.namedWindow('Video', cv.WINDOW_NORMAL)
try:
    cv.setWindowProperty('Video', cv.WND_PROP_TOPMOST, 1)
except cv.error:
    pass

def _normalize_features(feats_63: list[float], dropZ: bool) -> np.ndarray:
    if feats_63 is None or len(feats_63) != 63:
        return None
    wx, wy, wz = feats_63[0], feats_63[1], feats_63[2]
    centered = []
    for i in range(0, 63, 3):
        centered.extend([
            feats_63[i] - wx,
            feats_63[i + 1] - wy,
            feats_63[i + 2] - wz,
        ])
    max_dist = 0.0
    for i in range(0, 63, 3):
        dx, dy, dz = centered[i], centered[i + 1], centered[i + 2]
        d = (dx*dx + dy*dy + dz*dz) ** 0.5
        if d > max_dist:
            max_dist = d
    if max_dist > 1e-8:
        centered = [v / max_dist for v in centered]
    if dropZ:
        xy = []
        for i in range(0, 63, 3):
            xy.extend([centered[i], centered[i + 1]])
        return np.array(xy, dtype=np.float32)
    return np.array(centered, dtype=np.float32)

while True:
    success, frame = video.read()
    if not success:
        break
    cTime = time.time()
    fps = int (1 / (cTime - pTime))
    pTime = cTime
    cv.putText(frame, f"FPS: {fps}", (20, 80), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    raw = tracker.extractFeatures(frame, dropZ=False)
    feats = _normalize_features(raw, dropZ=dropZ)
    if feats is not None:
        X = feats.reshape(1, -1)
        pred = model.predict(X)[0]
        label = label_map.get(int(pred), str(pred))
        conf_txt = ""
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X).max()
            conf_txt = f" ({prob:.2f})"
        cv.putText(frame, f"{label_map[label]}{conf_txt}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
