#!/usr/bin/env bash
# 只装"告警图片/录像"静态服务（端口 8092）。
# 适合已经装过 mosquitto / Node-RED / ntfy 的板子，只补"告警截图与录像"这一件。
# 注意：若板上已有别的静态图片服务（例如占用 8082），两者互不影响，端口别冲突即可。
#
#   sudo bash deploy/install_alerts_http.sh
# 可用环境变量覆盖：PROJECT_DIR / ALERT_PORT / RUN_USER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$SCRIPT_DIR")}"
ALERT_PORT="${ALERT_PORT:-8092}"
RUN_USER="${RUN_USER:-firefly}"
ALERTS_DIR="${PROJECT_DIR}/alerts"

[ -d "$ALERTS_DIR" ] || mkdir -p "$ALERTS_DIR"
if ! id "$RUN_USER" >/dev/null 2>&1; then RUN_USER=root; fi
chown -R "$RUN_USER":"$RUN_USER" "$PROJECT_DIR" 2>/dev/null || true

echo "== 安装 safe-alerts-http.service =="
echo "   告警目录: $ALERTS_DIR"
echo "   端口:     $ALERT_PORT"
echo "   运行用户: $RUN_USER"

cat > /etc/systemd/system/safe-alerts-http.service <<EOF
[Unit]
Description=Serve SafeDetect alert images/videos for ntfy
After=network.target
[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_USER}
WorkingDirectory=${ALERTS_DIR}
ExecStart=/usr/bin/python3 -m http.server ${ALERT_PORT} --bind 0.0.0.0
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable safe-alerts-http
systemctl restart safe-alerts-http
sleep 1

echo "== 自检 =="
echo -n "  服务状态: "; systemctl is-active safe-alerts-http
if ss -ltn 2>/dev/null | grep -q ":${ALERT_PORT} "; then
  echo "  端口 ${ALERT_PORT}: 已监听"
else
  echo "  端口 ${ALERT_PORT}: **没监听**  -> journalctl -u safe-alerts-http -n 30"
fi

N=$(ls -1 "${ALERTS_DIR}"/*.jpg 2>/dev/null | wc -l)
echo "  alerts 目录里的 jpg 数量: ${N}"
if [ "${N}" -gt 0 ]; then
  F=$(ls -1 "${ALERTS_DIR}"/*.jpg | tail -1)
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${ALERT_PORT}/$(basename "$F")" 2>/dev/null || echo 000)
  echo "  取最近一张图 -> HTTP ${CODE} （期望 200）"
  echo "    http://127.0.0.1:${ALERT_PORT}/$(basename "$F")"
fi

echo
echo "手机通知里的截图/录像链接走的就是这个端口（${ALERT_PORT}）。"
echo "如果通知里的网址是 127.0.0.1 开头，手机打不开 —— 那是给 ntfy 本机抓附件用的；"
echo "给手机点开的链接用的是板子 IP（config/safe_config.json 的 notify.ts_ip，留空=自动探测）。"