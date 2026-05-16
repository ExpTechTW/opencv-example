import argparse
import random
import time

import cv2
import mediapipe as mp

from core.camera_utils import open_camera
from core.text_utils import ensure_font, put_text_cjk, text_size
from hand_direction import (
    build_landmarker,
    draw_hand,
    finger_states,
)

ROCK, PAPER, SCISSORS = "Rock", "Paper", "Scissors"
ZH = {ROCK: "石頭", PAPER: "布", SCISSORS: "剪刀"}

BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}

STATE_IDLE = "idle"
STATE_COUNTDOWN = "countdown"
STATE_RESULT = "result"


def classify_rps(states):
    """Rock      : 四指收 (拇指放寬)
       Paper     : 4 指打開且全部打開
       Scissors  : 食指 + 中指打開、無名指 + 小指收 (拇指放寬)
    """
    thumb, index, middle, ring, pinky = states
    if index and middle and not ring and not pinky:
        return SCISSORS
    if not index and not middle and not ring and not pinky:
        return ROCK
    if sum(states) >= 4 and index and middle and ring and pinky:
        return PAPER
    return None


def judge(player: str, comp: str) -> str:
    if player == comp:
        return "DRAW"
    if BEATS[player] == comp:
        return "WIN"
    return "LOSE"


def center_text(frame, text: str, y: int, size: int, color):
    w_text, _ = text_size(text, size)
    x = (frame.shape[1] - w_text) // 2
    put_text_cjk(frame, text, (x, y), size=size, color=color)


def run(use_csi: bool, cam_index: int):
    ensure_font()  # 預先下載字型
    cap = open_camera(use_csi=use_csi, index=cam_index)
    landmarker = build_landmarker()

    state = STATE_IDLE
    countdown_start = 0.0
    player_pick = None
    comp_pick = None
    result = None
    score = {"player": 0, "comp": 0, "draw": 0}

    t0 = time.time()
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

            live_pick = None
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                draw_hand(frame, lms)
                states = finger_states(lms, hd[0].category_name)
                live_pick = classify_rps(states) or live_pick

            # ===== 狀態機 =====
            if state == STATE_COUNTDOWN:
                elapsed = time.time() - countdown_start
                remain = 3 - elapsed
                if remain > 0:
                    n = int(remain) + 1 if remain > int(remain) else int(remain)
                    n = min(3, max(1, n))
                    center_text(frame, str(n), h // 2 - 80, 200, (0, 255, 255))
                    center_text(frame, "準備出拳", h // 2 + 130, 48, (255, 255, 255))
                else:
                    player_pick = live_pick
                    comp_pick = random.choice([ROCK, PAPER, SCISSORS])
                    if player_pick is None:
                        result = "NO_HAND"
                    else:
                        result = judge(player_pick, comp_pick)
                        if result == "WIN":
                            score["player"] += 1
                        elif result == "LOSE":
                            score["comp"] += 1
                        else:
                            score["draw"] += 1
                    state = STATE_RESULT
                    countdown_start = time.time()

            elif state == STATE_RESULT:
                box_w, box_h = 480, 200
                x0 = (w - box_w) // 2
                y0 = 40
                cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                              (0, 0, 0), -1)
                cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h),
                              (255, 255, 255), 2)
                p_label = ZH[player_pick] if player_pick else "未偵測"
                c_label = ZH[comp_pick] if comp_pick else "?"
                put_text_cjk(frame, f"你 : {p_label}", (x0 + 20, y0 + 15),
                             size=36, color=(0, 255, 0))
                put_text_cjk(frame, f"電腦 : {c_label}", (x0 + 20, y0 + 65),
                             size=36, color=(0, 200, 255))
                if result == "NO_HAND":
                    put_text_cjk(frame, "沒有偵測到手勢", (x0 + 20, y0 + 130),
                                 size=42, color=(0, 0, 255))
                else:
                    text_zh = {"WIN": "你贏了！", "LOSE": "你輸了", "DRAW": "平手"}[result]
                    color = {
                        "WIN": (0, 255, 0),
                        "LOSE": (0, 0, 255),
                        "DRAW": (255, 255, 0),
                    }[result]
                    put_text_cjk(frame, text_zh, (x0 + 20, y0 + 130),
                                 size=48, color=color)

            # ===== HUD =====
            put_text_cjk(frame,
                         f"你 {score['player']}   電腦 {score['comp']}   平手 {score['draw']}",
                         (10, 10), size=28, color=(255, 255, 255))

            if state == STATE_IDLE:
                put_text_cjk(frame, "按 空白鍵 開始,  Q 離開",
                             (10, h - 80), size=26, color=(0, 255, 255))
                if live_pick:
                    put_text_cjk(frame, f"目前手勢: {ZH[live_pick]}",
                                 (10, h - 40), size=26, color=(0, 200, 255))
            elif state == STATE_RESULT:
                put_text_cjk(frame, "空白鍵 = 下一局,  Q = 離開",
                             (10, h - 40), size=26, color=(0, 255, 255))
            elif state == STATE_COUNTDOWN and live_pick:
                put_text_cjk(frame, f"鎖定: {ZH[live_pick]}",
                             (10, h - 40), size=26, color=(0, 200, 255))

            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                if state in (STATE_IDLE, STATE_RESULT):
                    state = STATE_COUNTDOWN
                    countdown_start = time.time()
                    player_pick = comp_pick = result = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="剪刀石頭布")
    parser.add_argument("--csi", action="store_true", default=None, help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引或 sensor-id")
    args = parser.parse_args()
    use_csi = False if args.usb else args.csi
    run(use_csi, args.cam)
