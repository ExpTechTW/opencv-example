# OpenCV + MediaPipe 範例集

**OpenCV + MediaPipe** 實作的即時影像辨識小應用，包含手部追蹤、頭部姿態、表情辨識，以及兩個用手勢操作的小遊戲。在 **macOS / x86 Linux / Jetson Orin NX** 上都能跑。

## 功能

| 程式 | 功能 |
|---|---|
| `main.py` | 整合版：同時跑手部 + 頭部辨識，畫出 21 點手骨架與頭部姿態箭頭 |
| `hand_direction.py` | 偵測手部 21 個關鍵點，判斷食指指向方向（↑ ↓ ← →）與基本手勢（拳頭 / 張開 / 比讚 / 比 V…） |
| `head_direction.py` | 用臉部 mesh + PnP 解算頭部 Yaw / Pitch / Roll 三角度，輸出 `UP/DOWN/LEFT/RIGHT/CENTER` |
| `finger_count.py` | 數手指比的數字 0~10（兩手相加），需要拇指張開才會計入 |
| `rock_paper_scissors.py` | 剪刀石頭布小遊戲：SPACE 開始 → 3-2-1 倒數 → 比手勢 → 電腦隨機 → 全中文 UI |
| `emotion_detect.py` | 用 52 個 facial blendshapes 判斷 5 種表情（開心 / 難過 / 生氣 / 驚訝 / 平靜），對應 PNG 浮在頭頂上方 |

## 硬體 / 軟體需求

- **相機**：USB webcam 或 Jetson CSI 相機（IMX219 / IMX477 等）
- **Python**：3.8 / 3.10（Jetson）或 3.9-3.11（macOS / x86 Linux）
- **OS**：macOS、Ubuntu、Windows、JetPack 5 (Ubuntu 20.04) 或 JetPack 6 (Ubuntu 22.04)

> ⚠️ Python 3.12+ 上 MediaPipe 的 legacy solutions API 不可用，本專案已改用 Tasks API 避開這問題；但仍建議 ≤ 3.11 最穩定。

## 安裝

### macOS / x86 Linux

```bash
# 1. 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 2. 安裝依賴
pip install --upgrade pip
pip install -r requirements.txt

# 3. 跑跑看
python main.py
```

### Jetson Orin NX (16GB)

附 `install_jetson.sh` 一鍵搞定：

```bash
# 先安裝 venv 套件 (Ubuntu 預設沒裝)
sudo apt install -y python3.10-venv python3-opencv

# 跑安裝腳本
bash install_jetson.sh
```

腳本會做的事：

1. 建 `venv/`，全部依賴裝進去（不污染系統 Python）
2. 從 **PyPI 抓 mediapipe aarch64 wheel**（鎖 `<0.11`，因為 0.10.18 後不再出 aarch64）
3. 安裝 numpy<2、Pillow
4. **連結系統 OpenCV**（含 GStreamer + CUDA）進 venv，CSI 相機能用
5. 砍掉 mediapipe 連帶拉進來的 `opencv-contrib-python`（避免蓋掉系統版）
6. 驗證 + 輸出版本資訊

執行：
```bash
source venv/bin/activate
python main.py            # 自動偵測 Jetson → CSI 相機
python main.py --usb      # 強制用 USB 相機
```

效能最大化：
```bash
sudo nvpmodel -m 0      # MAXN 模式
sudo jetson_clocks      # 鎖最高頻率
```

## 使用

| 命令 | 說明 |
|---|---|
| `python main.py` | 手 + 頭部整合 |
| `python main.py --usb` | 在 Jetson 上強制用 USB 相機 |
| `python main.py --csi` | 在非 Jetson 上強制用 CSI 模式（測試 GStreamer） |
| `python main.py --cam 1` | 用第二顆相機（sensor-id=1 或 /dev/video1） |
| `python finger_count.py` | 比手指數字 |
| `python rock_paper_scissors.py` | SPACE 開局，Q 離開 |
| `python emotion_detect.py` | 表情辨識；`--debug` 顯示 blendshape 數值方便調閾值 |
| `python hand_direction.py` | 單獨手部 |
| `python head_direction.py` | 單獨頭部 |

**通用熱鍵**：
- `q`：離開
- `SPACE`：剪刀石頭布開局
- `d`：emotion_detect 切換 debug 數值

第一次執行時會自動下載：
- `models/hand_landmarker.task`（~5 MB）
- `models/face_landmarker.task`（~3 MB）
- `fonts/NotoSansTC-Regular.otf`（~21 MB，只有需要中文 UI 時才下）

下載後 cache 在本機，斷網也能跑。

## 專案結構

```
opencv-exmple/
├── core/                    ← 共用模組
│   ├── __init__.py
│   ├── camera_utils.py      ← USB / Jetson CSI 自動切換
│   ├── models.py            ← MediaPipe 模型自動下載
│   └── text_utils.py        ← OpenCV 上畫中文 (Noto Sans TC)
│
├── main.py                  ← 入口
├── hand_direction.py
├── head_direction.py
├── finger_count.py
├── rock_paper_scissors.py
├── emotion_detect.py
│
├── requirements.txt         ← macOS / x86 用
├── requirements-jetson.txt  ← Jetson 用
├── install_jetson.sh        ← Jetson 一鍵安裝
│
├── models/                  ← MediaPipe .task 模型 (自動下載)
├── fonts/                   ← Noto Sans TC 字型 (自動下載)
└── images/                  ← 5 個表情 PNG
```

## 演算法說明

### 手部方向（hand_direction.py）

- **拇指張開判定**：用「拇指 tip 到掌心中軸（wrist→middle MCP）的垂直距離」相對於掌寬 ≥ 50%
- **食指指向**：以 MCP(5) → TIP(8) 向量算 `atan2`，分四象限輸出方向
- **手勢分類**：根據 5 指 boolean state 分類 Fist / Open Palm / Pointing / Peace / Call Me / Thumb

### 頭部姿態（head_direction.py）

- 取 FaceMesh 6 個關鍵點（鼻尖、下巴、雙眼外角、嘴角）+ 預設 3D 人臉模型
- `cv2.solvePnP` 解 PnP → 旋轉矩陣 → 換算 Yaw / Pitch / Roll
- 用 ±12° 閾值輸出方向標籤

### 6-7 表情辨識（emotion_detect.py）

| 表情 | 判定條件 |
|---|---|
| **驚訝** | `jawOpen > 0.2` + `eyeWide > 0.08` + `browUp > 0.25` 三項任二觸發，且眉沒下壓 |
| **開心** | `mouthSmile > 0.45` |
| **生氣** | `browDown` 淨下壓 > 0.1 + （`mouthPress` 或 `eyeSquint` 高）|
| **難過** | `mouthFrown > 0.2` 或眉內側上抬且沒在笑沒張嘴 |
| **平靜** | 以上都不成立 |

最近 6 幀眾數平滑，避免單幀抖動。圖片定位以 FaceLandmark 10（額頭頂）為錨點，臉寬（234↔454）決定縮放。