#!/usr/bin/env bash
# SafeDetect_V0.01 便捷启动脚本（免敲一长串路径）
#
#   ./run.sh                                   # 用 config/safe_config.json 默认参数启动
#   ./run.sh --source /dev/video0 --noshow --alertdir alerts --stream 8090
#   ./run.sh --bench 30                        # 板端性能基准（不接摄像头）
#   ./run.sh --img test.jpg --out out.jpg      # 单张图调试
#
# 等价于：
#   python3 app/safedetect_two_stage_rk3568.py --config config/safe_config.json "$@"
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置：私有 safe_config.json 优先（真实 IP / 传感器接线 / 账号都填这里，不入库）；
# 刚 clone 下来还没有它时，退回可上传的公开模板 safe_config_git.json。
CFG="$DIR/config/safe_config.json"
[ -f "$CFG" ] || CFG="$DIR/config/safe_config_git.json"

exec python3 "$DIR/app/safedetect_two_stage_rk3568.py" --config "$CFG" "$@"
