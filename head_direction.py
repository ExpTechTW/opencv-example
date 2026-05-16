import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.camera_utils import open_camera
from core.models import ensure_model

LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)

CONTOUR_POINTS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


def head_pose(frame, landmarks):
    h, w = frame.shape[:2]
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS],
        dtype=np.float64,
    )

    focal = w
    cam_matrix = np.array([
        [focal, 0, w / 2],
        [0, focal, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, cam_matrix, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rot_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2]))
        yaw = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
        roll = np.degrees(np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1]))
        yaw = np.degrees(np.arctan2(-rot_matrix[2, 0], sy))
        roll = 0.0

    nose_end, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 80.0)]), rvec, tvec, cam_matrix, dist,
    )
    nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
    end_2d = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))
    return pitch, yaw, roll, nose_2d, end_2d


def direction_label(pitch: float, yaw: float, threshold: float = 12.0) -> str:
    p = pitch
    if p > 90:
        p -= 180
    elif p < -90:
        p += 180

    horiz = "CENTER"
    vert = ""
    if yaw > threshold:
        horiz = "LEFT"
    elif yaw < -threshold:
        horiz = "RIGHT"
    if p > threshold:
        vert = "DOWN"
    elif p < -threshold:
        vert = "UP"

    if vert and horiz != "CENTER":
        return f"{vert}-{horiz}"
    if vert:
        return vert
    return horiz


def draw_face_contour(frame, landmarks):
    h, w = frame.shape[:2]
    for idx in CONTOUR_POINTS:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def build_landmarker():
    model_path = ensure_model("face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


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

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                draw_face_contour(frame, lm)
                pose = head_pose(frame, lm)
                if pose is not None:
                    pitch, yaw, roll, nose_2d, end_2d = pose
                    direction = direction_label(pitch, yaw)
                    cv2.arrowedLine(frame, nose_2d, end_2d, (0, 255, 0), 3, tipLength=0.2)
                    cv2.putText(frame, f"Direction: {direction}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.putText(frame, f"Yaw: {yaw:+.1f}  Pitch: {pitch:+.1f}  Roll: {roll:+.1f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Head Direction (q to quit)", frame)
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
