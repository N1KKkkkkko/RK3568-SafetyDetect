#!/usr/bin/env bash
# SafeDetect_V0.01 便捷启动脚本（免敲一长串路径）
#
#   ./run.sh                                   # 用 config/safe_config.json 默认参数启动
#   ./run.sh --source /dev/video0 --noshow --alertdir alerts --stream 8090
#   ./run.sh --bench 30                        # 板端性能基准（不接摄像头）
#   ./run.sh --img test.jpg --out out.jpg      # 单张图调试
#
# 送测视频文件（路径按当前目录解析；视频按自身帧率循环播放，Ctrl+C 退出）：
#   ./run.sh --source test2.mp4 --noshow                       # 直接回放
#   ./run.sh --source test2.mp4 --noshow --stream 0            # 回放，不开 MJPEG 预览端口
#   ./run.sh --source test2.mp4 --out-video out.mp4 --noshow   # 回放并把标注画面另存为视频
#   ./run.sh --source /abs/path/test2.mp4 --noshow             # 视频不在当前目录时用绝对路径
#
# 等价于：
#   python3 app/safedetect_rk3568.py --config config/safe_config.json "$@"
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置：私有 safe_config.json 优先（真实 IP / 传感器接线 / 账号都填这里，不入库）；
# 刚 clone 下来还没有它时，退回可上传的公开模板 safe_config_git.json。
CFG="$DIR/config/safe_config.json"
[ -f "$CFG" ] || CFG="$DIR/config/safe_config_git.json"

exec python3 "$DIR/app/safedetect_rk3568.py" --config "$CFG" "$@"
