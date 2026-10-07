#!/usr/bin/env bash
# SafeDetect_V0.01 —— PC 端送测脚本（离线，走 onnxruntime，不需要板子 / NPU）
#
# 和板端 run.sh 的区别：
#   run.sh     板端完整程序：rknnlite + NPU 推理 + 摄像头 + 推流 + MQTT 告警，
#              必须跑在 RK3568 上，需要摄像头节点和板端运行时。
#   run_pc.sh  开发机离线送测：headcut ONNX + onnxruntime，跑的是同一套解码 / 装备归属 /
#              状态机逻辑，不需要 NPU，用来在电脑上看效果、调阈值、批量出报告。
#
#   ./run_pc.sh --img test.jpg --out-dir /tmp/pc_out      # 单张图
#   ./run_pc.sh --dir ~/frames --out-dir /tmp/pc_out      # 批量图片（出图 + CSV）
#   ./run_pc.sh --video test2.mp4 --out-video out2.mp4    # 视频：输出标注视频
#   ./run_pc.sh --video test2.mp4 --out-video out2.mp4 --out-csv out2.csv
#   ./run_pc.sh --video test2.mp4 --max-frames 100 --out-video out2.mp4   # 只跑前 100 帧
#   ./run_pc.sh --video test2.mp4 --confirm-frames 5 --out-video out2.mp4 # 连续 5 帧才告警
#
# 环境变量（都可选）：
#   PYTHON  指定带 cv2 + onnxruntime 的 python（默认自动找 ../../venv310）
#   ONNX    指定 headcut ONNX（默认 models/own_best_headcut.onnx）
#
# 注意：这里跑的是 fp32 的 ONNX，与板端 i8 的 rknn 会有少量量化差异，
#       最终数值仍要上板回归；但解码/归属/状态机是同一份代码，阈值可直接互搬。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 找一个装了 cv2 + onnxruntime 的解释器
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in "$DIR/../../venv310/bin/python" "$DIR/../venv310/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import cv2, onnxruntime" >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "找不到带 cv2 + onnxruntime 的 Python。"
  echo "  方法一：指定解释器，例如  PYTHON=~/venv310/bin/python ./run_pc.sh ..."
  echo "  方法二：pip install onnxruntime opencv-python numpy"
  exit 1
fi

ONNX="${ONNX:-$DIR/models/own_best_headcut.onnx}"
if [ ! -f "$ONNX" ]; then
  echo "找不到 ONNX 模型: $ONNX"
  echo "PC 送测用的是 headcut ONNX（models/own_best_headcut.onnx），不是 rknn。"
  exit 1
fi

echo "PC 送测  python=$PY"
echo "         onnx=$ONNX"
exec "$PY" "$DIR/tools/pc_onnx_check.py" --onnx "$ONNX" "$@"
