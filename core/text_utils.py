import os
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 從 core/text_utils.py 往上一層 → 專案根, 字型 cache 放在專案根的 fonts/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(_PROJECT_ROOT, "fonts")
FONT_FILE = "NotoSansTC-Regular.otf"
FONT_URL = (
    "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/"
    "TraditionalChinese/NotoSansCJKtc-Regular.otf"
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def ensure_font() -> str:
    os.makedirs(FONT_DIR, exist_ok=True)
    path = os.path.join(FONT_DIR, FONT_FILE)
    if not os.path.exists(path):
        print(f"[download] {FONT_FILE} ...")
        urllib.request.urlretrieve(FONT_URL, path)
        print(f"[done] -> {path}")
    return path


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(ensure_font(), size)
    return _FONT_CACHE[size]


def put_text_cjk(frame, text: str, org, size: int = 24,
                 color=(255, 255, 255)) -> np.ndarray:
    """在 BGR 影像上畫中文。color 是 BGR。"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    rgb = (color[2], color[1], color[0])
    draw.text(org, text, font=_font(size), fill=rgb)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def text_size(text: str, size: int = 24) -> tuple[int, int]:
    bbox = _font(size).getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]
