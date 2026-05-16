import os

import cv2


def is_jetson() -> bool:
    """偵測是否在 NVIDIA Jetson 上執行 (任何型號)。"""
    return os.path.exists("/etc/nv_tegra_release")


def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def open_camera(use_csi=None, index: int = 0, width: int = 960, height: int = 540) -> cv2.VideoCapture:
    """開啟相機。
    use_csi:
        True  → 強制使用 CSI (nvarguscamerasrc, 限 Jetson)
        False → 強制使用 USB / V4L2
        None  → 自動: 在 Jetson 上預設 CSI, 其他平台 USB
    """
    if use_csi is None:
        use_csi = is_jetson()

    if use_csi:
        cap = cv2.VideoCapture(
            gstreamer_pipeline(sensor_id=index, display_width=width, display_height=height),
            cv2.CAP_GSTREAMER,
        )
    else:
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        mode = "CSI (nvarguscamerasrc)" if use_csi else "USB"
        raise RuntimeError(f"無法開啟相機 [{mode}] index={index}. 請確認硬體或改用 --usb / --csi 強制切換。")
    return cap
