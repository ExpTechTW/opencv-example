import argparse
import math
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.camera_utils import open_camera
from core.models import ensure_model

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def thumb_extended(landmarks) -> bool:
    # 計算 thumb tip(4) 到「掌心中軸 (wrist 0 → middle MCP 9)」的垂直距離，
    # 以掌寬 (index MCP 5 ↔ pinky MCP 17) 當基準。
    # 拇指張開向外 → tip 偏離中軸 → perp 大
    # 拇指收進掌心 → tip 落在中軸附近 → perp 小
    w = landmarks[0]
    m = landmarks[9]
    t = landmarks[4]
    dx, dy = m.x - w.x, m.y - w.y
    axis_len = math.hypot(dx, dy)
    if axis_len == 0:
        return False
    perp = abs((t.x - w.x) * dy - (t.y - w.y) * dx) / axis_len
    palm_width = _dist(landmarks[5], landmarks[17])
    if palm_width == 0:
        return False
    return perp > palm_width * 0.5


def finger_states(landmarks, handedness_label: str):
    states = [thumb_extended(landmarks)]
    for tip, pip in zip(TIP_IDS[1:], PIP_IDS[1:]):
        states.append(landmarks[tip].y < landmarks[pip].y)
    return states


def pointing_direction(landmarks) -> str:
    base = landmarks[5]
    tip = landmarks[8]
    dx = tip.x - base.x
    dy = tip.y - base.y
    angle = math.degrees(math.atan2(-dy, dx))
    if -45 <= angle < 45:
        return "RIGHT →"
    if 45 <= angle < 135:
        return "UP ↑"
    if angle >= 135 or angle < -135:
        return "LEFT ←"
    return "DOWN ↓"


def classify_gesture(states) -> str:
    thumb, index, middle, ring, pinky = states
    if not any(states):
        return "Fist"
    if all(states):
        return "Open Palm"
    if index and not middle and not ring and not pinky:
        return "Pointing"
    if index and middle and not ring and not pinky:
        return "Peace"
    if thumb and pinky and not index and not middle and not ring:
        return "Call Me"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumb"
    return "Unknown"


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)


def build_landmarker():
    model_path = ensure_model("hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def run(use_csi: bool, cam_index: int):
    cap = open_camera(use_csi=use_csi, index=cam_index)
    landmarker = build_landmarker()
    t0 = time.time()
    prev_t = cv2.getTickCount()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            for lms, hd in zip(result.hand_landmarks, result.handedness):
                label = hd[0].category_name
                states = finger_states(lms, label)
                gesture = classify_gesture(states)
                direction = pointing_direction(lms) if states[1] else "-"
                draw_hand(frame, lms)
                h, w = frame.shape[:2]
                x = int(lms[0].x * w)
                y = int(lms[0].y * h)
                cv2.putText(frame, f"{label} | {gesture}", (x - 30, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Dir: {direction}", (x - 30, y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Hand Direction (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi", action="store_true", default=None, help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引或 sensor-id")
    args = parser.parse_args()
    use_csi = False if args.usb else args.csi
    run(use_csi, args.cam)
