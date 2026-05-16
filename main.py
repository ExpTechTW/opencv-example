import argparse
import time

import cv2
import mediapipe as mp

from core.camera_utils import open_camera
from hand_direction import (
    build_landmarker as build_hand,
    classify_gesture,
    draw_hand,
    finger_states,
    pointing_direction,
)
from head_direction import (
    build_landmarker as build_face,
    direction_label,
    draw_face_contour,
    head_pose,
)


def run(use_csi: bool, cam_index: int):
    cap = open_camera(use_csi=use_csi, index=cam_index)
    hand_lm = build_hand()
    face_lm = build_face()

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

            hand_res = hand_lm.detect_for_video(mp_image, ts_ms)
            face_res = face_lm.detect_for_video(mp_image, ts_ms)

            for lms, hd in zip(hand_res.hand_landmarks, hand_res.handedness):
                label = hd[0].category_name
                states = finger_states(lms, label)
                gesture = classify_gesture(states)
                direction = pointing_direction(lms) if states[1] else "-"
                draw_hand(frame, lms)
                h, w = frame.shape[:2]
                x = int(lms[0].x * w)
                y = int(lms[0].y * h)
                cv2.putText(frame, f"{label}: {gesture}", (x - 30, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Point: {direction}", (x - 30, y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            if face_res.face_landmarks:
                lm = face_res.face_landmarks[0]
                draw_face_contour(frame, lm)
                pose = head_pose(frame, lm)
                if pose is not None:
                    pitch, yaw, roll, nose_2d, end_2d = pose
                    direction = direction_label(pitch, yaw)
                    cv2.arrowedLine(frame, nose_2d, end_2d, (0, 255, 0), 3, tipLength=0.2)
                    cv2.putText(frame, f"Head: {direction}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    cv2.putText(frame, f"Yaw:{yaw:+.0f} Pitch:{pitch:+.0f} Roll:{roll:+.0f}",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Jetson Orin NX - Hand + Head Direction (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_lm.close()
        face_lm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Orin NX 手指 + 頭部方向辨識")
    parser.add_argument("--csi", action="store_true", default=None, help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引或 sensor-id (預設 0)")
    args = parser.parse_args()
    use_csi = False if args.usb else args.csi
    run(use_csi, args.cam)
