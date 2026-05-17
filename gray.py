"""灰階化工具:
  即時相機:  python gray.py
  轉檔:     python gray.py --image photo.jpg [--out photo_gray.jpg]

熱鍵:
  q  離開
  g  切換灰階 / 彩色
  s  存目前畫面 → snapshot_時間戳.png
"""
import argparse
import os
import time

import cv2

from core.camera_utils import open_camera


def gray_image(path: str, out: str | None = None) -> str:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"讀不到圖片: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if out is None:
        base, ext = os.path.splitext(path)
        out = f"{base}_gray{ext or '.png'}"
    cv2.imwrite(out, gray)
    print(f"[OK] {path}  →  {out}  ({gray.shape[1]}x{gray.shape[0]})")
    return out


def gray_camera(use_csi, cam_index: int):
    cap = open_camera(use_csi=use_csi, index=cam_index)
    show_gray = True
    prev_t = cv2.getTickCount()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            if show_gray:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 為了能在上面畫彩色文字, 轉回 3-channel BGR (數值仍是灰階)
                disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                disp = frame

            curr_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (curr_t - prev_t)
            prev_t = curr_t
            mode = "GRAY" if show_gray else "COLOR"
            cv2.putText(disp, f"FPS:{fps:.1f}  [{mode}]   g=toggle  s=save  q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Grayscale", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('g'):
                show_gray = not show_gray
            if key == ord('s'):
                fname = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fname, disp)
                print(f"[saved] {fname}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="灰階化 (圖片檔 / 即時相機)")
    parser.add_argument("--image", "-i", help="要灰階化的圖片路徑 (給了就只轉檔, 不開相機)")
    parser.add_argument("--out", "-o", help="輸出檔名 (預設: 原檔名_gray.ext)")
    parser.add_argument("--csi", action="store_true", default=None,
                        help="強制 CSI 相機 (Jetson 預設)")
    parser.add_argument("--usb", action="store_true", help="強制 USB 相機")
    parser.add_argument("--cam", type=int, default=0, help="相機索引 / sensor-id")
    args = parser.parse_args()

    if args.image:
        gray_image(args.image, args.out)
    else:
        use_csi = False if args.usb else args.csi
        gray_camera(use_csi, args.cam)
