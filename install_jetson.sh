#!/usr/bin/env bash
# Jetson Orin NX / Orin Nano / Xavier / Nano 自動安裝腳本
# 預設行為: 建立 venv, 借用系統 OpenCV (含 GStreamer, CSI 相機可用)
# 用法:
#   bash install_jetson.sh             # 預設: 系統 OpenCV (CSI + USB 都能用)
#   bash install_jetson.sh --pip-cv    # 用 pip OpenCV (只 USB, 但版本較新)
set -e

# ===== 0. 切到腳本所在目錄 (從哪執行都 OK) =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[info] 工作目錄: $SCRIPT_DIR"

# ===== 1. 參數 =====
USE_SYS_CV=1
for arg in "$@"; do
    case "$arg" in
        --pip-cv|--no-csi) USE_SYS_CV=0 ;;
        --help|-h)
            sed -n '2,7p' "$0"
            exit 0 ;;
        *) echo "[ERROR] 不認得參數: $arg" ; exit 1 ;;
    esac
done

# ===== 2. 已在另一個 venv 中? =====
if [ -n "$VIRTUAL_ENV" ]; then
    echo "[ERROR] 偵測到目前已在 venv 中: $VIRTUAL_ENV"
    echo "        請先執行: deactivate 然後再跑這支腳本"
    exit 1
fi

# ===== 3. 架構檢查 =====
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "[ERROR] 這支腳本是給 Jetson (aarch64) 用的, 你的架構是 $ARCH"
    echo "        macOS / x86 Linux 請用: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -f /etc/nv_tegra_release ]; then
    echo "[WARN] 找不到 /etc/nv_tegra_release, 看起來不是 Jetson? 仍嘗試繼續..."
else
    L4T=$(head -1 /etc/nv_tegra_release | grep -oE 'R[0-9]+' | head -1)
    echo "[info] L4T 版本: $L4T"
fi

# ===== 4. Python 版本檢查 =====
PY_VER=$(python3 -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
PY_CP=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
echo "[info] 系統 Python: $PY_VER ($PY_CP)"

# mediapipe 在 PyPI 上對 aarch64 + cp38 / cp310 都有 wheel (到 0.10.18 為止)
# 之後版本作者停止出 aarch64 wheel, 所以鎖 <0.11
case "$PY_CP" in
    cp38|cp310) ;;
    *)
        echo "[ERROR] Python $PY_VER 在 Jetson 上 mediapipe 沒有官方 wheel"
        echo "        支援: Python 3.8 (JetPack 5), Python 3.10 (JetPack 6)"
        exit 1 ;;
esac

if [ $USE_SYS_CV -eq 1 ]; then
    echo "[info] OpenCV: 系統版 (含 GStreamer, CSI + USB 都能用)"
else
    echo "[info] OpenCV: pip 版 (只能 USB, 版本較新)"
fi
echo ""

# ===== 5. 建 venv =====
VENV_DIR="$SCRIPT_DIR/venv"

# 預檢: python3-venv 套件 (Ubuntu/Debian 預設沒裝)
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    echo "[ERROR] 缺少 python${PY_VER}-venv 套件 (ensurepip 不可用)"
    echo "        請執行:"
    echo "            sudo apt install -y python${PY_VER}-venv"
    echo "        然後重跑這支腳本。"
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    # 完整性檢查: 上次建到一半的話 bin/activate 會缺
    if [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "[info] 既有 venv 不完整 (缺 bin/activate 或 bin/python) → 重建"
        rm -rf "$VENV_DIR"
    else
        EXISTING_PY=$("$VENV_DIR/bin/python" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "?")
        if [ "$EXISTING_PY" != "$PY_VER" ]; then
            echo "[info] 既有 venv 是 Python $EXISTING_PY, 與目標 $PY_VER 不符 → 重建"
            rm -rf "$VENV_DIR"
        else
            echo "[info] 沿用既有 venv ($VENV_DIR)"
        fi
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[step] 建立 venv → $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# 從這裡開始所有 pip/python 都在 venv 內
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[info] 已啟用 venv: $VIRTUAL_ENV"

pip install --upgrade pip wheel setuptools

# ===== 6. 裝 mediapipe + 其他依賴 =====
# 鎖 mediapipe <0.11 因為 0.10.18 之後的版本 PyPI 不再出 aarch64 wheel
echo "[step] 安裝 mediapipe (PyPI 上有 aarch64 wheel) 與其他依賴 ..."
pip install -r "$SCRIPT_DIR/requirements-jetson.txt"

# mediapipe 會把 opencv-contrib-python 當依賴拉進來, 會蓋掉系統 OpenCV
# 預先全砍, 後面再決定要不要裝 pip 版
echo "[step] 移除 mediapipe 拉進來的 pip OpenCV (避免蓋掉系統版) ..."
pip uninstall -y \
    opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true

# ===== 8. OpenCV 處理 =====
if [ $USE_SYS_CV -eq 1 ]; then
    echo "[step] 連結系統 OpenCV (含 GStreamer) 到 venv ..."
    SYS_SITE="/usr/lib/python${PY_VER}/dist-packages"
    if [ ! -d "$SYS_SITE" ]; then
        echo "[ERROR] 找不到 $SYS_SITE, 系統 OpenCV 可能未安裝"
        echo "        試試: sudo apt install python3-opencv"
        exit 1
    fi
    VENV_SITE="$VENV_DIR/lib/python${PY_VER}/site-packages"
    echo "$SYS_SITE" > "$VENV_SITE/system.pth"
else
    echo "[step] 安裝 pip 版 opencv-python ..."
    pip install "opencv-python>=4.8.0"
fi

# ===== 9. 驗證 =====
echo ""
echo "[step] 驗證安裝 ..."
python - <<'PY'
import mediapipe as mp
import cv2, numpy, PIL
print(f"  mediapipe : {mp.__version__}")
print(f"  opencv    : {cv2.__version__}")
print(f"  numpy     : {numpy.__version__}")
print(f"  Pillow    : {PIL.__version__}")
info = cv2.getBuildInformation()
for keyword in ("GStreamer", "CUDA"):
    line = next((l for l in info.splitlines() if keyword in l), "")
    print(f"  cv2.{keyword:10s}: {line.strip()}")
PY

# ===== 10. 完成提示 =====
echo ""
echo "================================================================"
echo "[OK] 安裝完成"
echo "================================================================"
echo "  venv 路徑: $VENV_DIR"
echo "  啟動方式: source venv/bin/activate"
echo ""
echo "執行 (在 venv 內):"
echo "  python main.py                   # 手 + 頭, 自動用 CSI 相機"
echo "  python main.py --usb             # 改用 USB 相機"
echo "  python finger_count.py           # 數手指"
echo "  python rock_paper_scissors.py    # 剪刀石頭布"
echo "  python emotion_detect.py         # 表情偵測"
echo ""
echo "Jetson 效能最大化 (選用):"
echo "  sudo nvpmodel -m 0      # 解鎖 MAXN 模式"
echo "  sudo jetson_clocks      # 鎖最高頻率"
