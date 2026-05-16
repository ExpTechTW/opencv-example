import os
import urllib.request

# 從 core/models.py 往上一層 → 專案根, 模型 cache 放在專案根的 models/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}


def ensure_model(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"未知模型: {name}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"[download] {name} ...")
        urllib.request.urlretrieve(MODELS[name], path)
        print(f"[done] -> {path}")
    return path
