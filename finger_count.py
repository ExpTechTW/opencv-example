import argparse
import time

import cv2
import mediapipe as mp

from core.camera_utils import open_camera
from hand_direction import (
    build_landmarker,
    draw_hand,
    finger_states,
)

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def count_fingers(states) -> int:
    return sum(1 for s in states if s)


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

            total = 0
            h, w = frame.shape[:2]
            for lms, hd in zip(result.hand_landmarks, result.handedness):
                label = hd[0].category_name
                states = finger_states(lms, label)
                n = count_fingers(states)
                total += n

                draw_hand(frame, lms)
                up = ",".join(FINGER_NAMES[i] for i, s in enumerate(states) if s) or "None"
                x = int(lms[0].x * w)
                y = int(lms[0].y * h)
                cv2.putText(frame, f"{label}: {n}", (x - 30, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, up, (x - 30, y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            # 大字顯示總數
            cv2.rectangle(frame, (w - 180, 10), (w - 10, 130), (0, 0, 0), -1)
            cv2.putText(frame, "Total", (w - 165, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, str(total), (w - 150, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Finger Count (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="偵測手指比的數字 (0-10)")
    parser.add_argument("--csi", action="store_true", default=None, help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引或 sensor-id")
    args = parser.parse_args()
    use_csi = False if args.usb else args.csi
    run(use_csi, args.cam)
