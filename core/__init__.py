"""共用模組:
    camera_utils — 相機開啟 (USB / Jetson CSI 自動偵測)
    models       — MediaPipe 模型自動下載
    text_utils   — OpenCV 影像上畫中文 (Noto Sans TC)

便利的頂層 import:
    from core import open_camera, ensure_model, put_text_cjk
"""
from .camera_utils import gstreamer_pipeline, is_jetson, open_camera
from .models import ensure_model
from .text_utils import ensure_font, put_text_cjk, text_size

__all__ = [
    "gstreamer_pipeline",
    "is_jetson",
    "open_camera",
    "ensure_model",
    "ensure_font",
    "put_text_cjk",
    "text_size",
]
