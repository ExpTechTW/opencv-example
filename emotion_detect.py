import argparse
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from core.camera_utils import open_camera
from core.models import ensure_model
from core.text_utils import ensure_font, put_text_cjk

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral"]
EMOTION_ZH = {
    "happy": "開心",
    "sad": "難過",
    "angry": "生氣",
    "surprise": "驚訝",
    "neutral": "平靜",
}


def load_emotion_images(size: int = 260):
    imgs = {}
    for name in EMOTIONS:
        path = os.path.join(IMAGES_DIR, f"{name}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] 找不到 {path}")
            continue
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        if img.shape[2] == 3:
            alpha = np.full((nh, nw, 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        imgs[name] = img
    return imgs


def overlay_bgra(bg, fg, x, y):
    h, w = fg.shape[:2]
    H, W = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    fx0, fy0 = x0 - x, y0 - y
    fx1, fy1 = fx0 + (x1 - x0), fy0 + (y1 - y0)
    alpha = fg[fy0:fy1, fx0:fx1, 3:4].astype(np.float32) / 255.0
    bg[y0:y1, x0:x1] = (1 - alpha) * bg[y0:y1, x0:x1] + alpha * fg[fy0:fy1, fx0:fx1, :3]


def bs_score(blendshapes, name: str) -> float:
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def classify_emotion(bs):
    """以 ARKit blendshape (對應 FACS Action Units) 規則判斷表情。
    驚訝 vs 生氣 的關鍵差異在「眉毛上抬 (AU1/2) vs 眉毛下壓 (AU4)」與
    「下顎張開 (AU26) vs 抿嘴 + 眼緊瞇 (AU24+AU7)」。
    """
    smile = (bs_score(bs, "mouthSmileLeft") + bs_score(bs, "mouthSmileRight")) / 2
    frown = (bs_score(bs, "mouthFrownLeft") + bs_score(bs, "mouthFrownRight")) / 2
    jaw_open = bs_score(bs, "jawOpen")
    eye_wide = (bs_score(bs, "eyeWideLeft") + bs_score(bs, "eyeWideRight")) / 2
    eye_squint = (bs_score(bs, "eyeSquintLeft") + bs_score(bs, "eyeSquintRight")) / 2
    brow_inner = bs_score(bs, "browInnerUp")
    brow_outer = (bs_score(bs, "browOuterUpLeft") + bs_score(bs, "browOuterUpRight")) / 2
    brow_down = (bs_score(bs, "browDownLeft") + bs_score(bs, "browDownRight")) / 2
    mouth_press = (bs_score(bs, "mouthPressLeft") + bs_score(bs, "mouthPressRight")) / 2

    brow_up = max(brow_inner, brow_outer)
    brow_net = brow_up - brow_down  # 正 = 眉上抬, 負 = 眉下壓

    debug = {
        "smile": smile, "frown": frown, "jaw": jaw_open,
        "eyeW": eye_wide, "eyeSq": eye_squint,
        "browU": brow_up, "browD": brow_down, "browNet": brow_net,
        "press": mouth_press,
    }

    # 1) 驚訝: 三個訊號 (嘴開、眼睜、眉上抬) 任兩個觸發, 且眉沒下壓。
    # eyeWide blendshape 偏難達標, 所以用「投票制」靠 brow_up 補強。
    mouth_open = jaw_open > 0.2
    eye_wide_open = eye_wide > 0.08
    brow_lift = brow_up > 0.25
    if brow_down < 0.3 and sum([mouth_open, eye_wide_open, brow_lift]) >= 2:
        return "surprise", debug

    # 2) 開心: 嘴角明顯上揚
    if smile > 0.4:
        return "happy", debug

    # 3) 生氣: 眉淨下壓 + (抿嘴 或 眼緊瞇)。互斥條件: 眉必須是下壓狀態。
    if brow_net < -0.1 and brow_down > 0.3 and (mouth_press > 0.2 or eye_squint > 0.25):
        return "angry", debug

    # 4) 難過: 嘴角下垂, 或 (眉內側上抬 + 沒笑 + 沒張嘴)
    if frown > 0.2 or (brow_inner > 0.35 and smile < 0.15 and jaw_open < 0.15):
        return "sad", debug

    return "neutral", debug


def build_landmarker():
    model_path = ensure_model("face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def run(use_csi: bool, cam_index: int, show_debug: bool):
    ensure_font()
    cap = open_camera(use_csi=use_csi, index=cam_index)
    landmarker = build_landmarker()
    images = load_emotion_images(size=400)  # 來源圖, 後續會依臉寬縮放
    history = deque(maxlen=6)  # 多數決平滑

    t0 = time.time()
    prev_t = cv2.getTickCount()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            res = landmarker.detect_for_video(mp_image, ts_ms)

            current = None
            scores = {}
            if res.face_blendshapes:
                current, scores = classify_emotion(res.face_blendshapes[0])
                history.append(current)

            display = (Counter(history).most_common(1)[0][0]
                       if history else "neutral")

            # 將表情圖疊到頭部上方
            if display in images and res.face_landmarks:
                lm = res.face_landmarks[0]
                head_x = int(lm[10].x * w)   # 額頭頂中心
                head_y = int(lm[10].y * h)
                face_w_px = abs(lm[454].x - lm[234].x) * w  # 兩側臉頰寬

                target_w = max(60, int(face_w_px * 1.1))
                src = images[display]
                scale = target_w / src.shape[1]
                new_w = max(1, int(src.shape[1] * scale))
                new_h = max(1, int(src.shape[0] * scale))
                img_r = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)

                ix = head_x - new_w // 2
                iy = head_y - new_h - 15  # 頭頂上方 15px
                overlay_bgra(frame, img_r, ix, iy)

            put_text_cjk(frame, f"目前表情: {EMOTION_ZH.get(display, '?')}",
                         (10, 10), size=32, color=(0, 255, 255))

            if show_debug and scores:
                y_off = 60
                for k, v in scores.items():
                    txt = f"{k:6s}: {v:.2f}"
                    cv2.putText(frame, txt, (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 200, 200), 1)
                    y_off += 22

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Emotion (q to quit, d toggles debug)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('d'):
                show_debug = not show_debug
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="臉部表情辨識 + 對應圖片")
    parser.add_argument("--csi", action="store_true", default=None, help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引或 sensor-id")
    parser.add_argument("--debug", action="store_true", help="顯示 blendshape 數值")
    args = parser.parse_args()
    use_csi = False if args.usb else args.csi
    run(use_csi, args.cam, args.debug)
