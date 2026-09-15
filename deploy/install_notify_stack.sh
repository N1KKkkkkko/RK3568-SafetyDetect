#!/usr/bin/env bash
# SafeDetect_V0.01 - one-shot setup on RK3568 board
# 端口约定：
#   本包使用 8090(MJPEG 实时预览) / 8092(告警截图与录像的静态 HTTP)。
#   MQTT 1883、Node-RED 1880、ntfy 8081 属于板子级共用服务，本脚本重复执行是幂等的；
#   mosquitto (MQTT broker) + Node-RED (notify hub) + ntfy (phone push)
#   + safe-alerts-http (serves SafeDetect alert images/videos, port 8092)
# Run on the board:
#   sudo bash deploy/install_notify_stack.sh
# Config (env vars, optional):
#   PROJECT_DIR - where SafeDetect_V0.01 is placed on the board
#   CAMERA_DEV  - USB camera video node
#   NTFY_VER    - ntfy release version (linux_arm64)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

PROJECT_DIR=${PROJECT_DIR:-/home/firefly/rk3568_camera/SafeDetect_V0.01}
ALERTS_DIR=${PROJECT_DIR}/alerts
CAMERA_DEV=${CAMERA_DEV:-/dev/safecam}
NTFY_VER=${NTFY_VER:-2.11.0}
BOARD_IP=$(hostname -I 2>/dev/null | awk '{print $1}')

if [ "$(uname -m)" != "aarch64" ]; then
  echo "WARN: this script is written for aarch64 (RK3568), got $(uname -m)"
fi

echo "== [1/6] mosquitto =="
apt-get update
apt-get install -y mosquitto mosquitto-clients wget tar
cat > /etc/mosquitto/conf.d/notify.conf <<EOF
listener 1883 0.0.0.0
allow_anonymous true
EOF
systemctl enable mosquitto
systemctl restart mosquitto

echo "== [2/6] Node.js + Node-RED =="
# Ubuntu 20.04 自带 nodejs 太老，Node-RED 需要 >=14，用 NodeSource 装 Node 20 LTS
if ! command -v node >/dev/null 2>&1 || [ "$(node -e 'console.log(process.versions.node.split(".")[0])')" -lt 14 ]; then
  apt-get install -y curl
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
if ! command -v node-red >/dev/null 2>&1; then
  # npm 国内很慢，换 npmmirror 镜像；root 装全局包必须 --unsafe-perm
  npm config set registry https://registry.npmmirror.com 2>/dev/null || true
  echo "== installing node-red (can take a few minutes, please wait) =="
  npm install -g --unsafe-perm --no-audit --no-fund node-red
fi
cat > /etc/systemd/system/node-red.service <<'EOF'
[Unit]
Description=Node-RED
After=mosquitto.service
[Service]
Type=simple
User=firefly
Group=firefly
ExecStart=/usr/bin/env node-red
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable node-red
systemctl restart node-red

echo "== [3/6] ntfy =="
if ! command -v ntfy >/dev/null 2>&1; then
  cd /tmp
  wget -q https://github.com/binwiederhier/ntfy/releases/download/v${NTFY_VER}/ntfy_${NTFY_VER}_linux_arm64.tar.gz
  tar -xzf ntfy_${NTFY_VER}_linux_arm64.tar.gz
  cp ntfy_${NTFY_VER}_linux_arm64/ntfy /usr/bin/ntfy
  chmod +x /usr/bin/ntfy
fi
mkdir -p /etc/ntfy /var/cache/ntfy /var/lib/ntfy
cat > /etc/ntfy/server.yml <<EOF
base-url: http://${BOARD_IP}:8081
listen-http: :8081
cache-file: /var/cache/ntfy/cache.db
cache-duration: 12h
attachment-cache-dir: /var/cache/ntfy/attachments
auth-file: /var/lib/ntfy/user.db
auth-default-access: read-write
EOF
cat > /etc/systemd/system/ntfy.service <<'EOF'
[Unit]
Description=ntfy push server
After=network.target
[Service]
Type=simple
ExecStart=/usr/bin/ntfy serve
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable ntfy
systemctl restart ntfy

echo "== [4/6] alert image HTTP server =="
mkdir -p ${ALERTS_DIR}
# services run as firefly, so the project dir must be writable by firefly
chown -R firefly:firefly ${PROJECT_DIR}
cat > /etc/systemd/system/safe-alerts-http.service <<EOF
[Unit]
Description=serve SafeDetect alert images/videos for ntfy
After=network.target
[Service]
Type=simple
User=firefly
Group=firefly
WorkingDirectory=${ALERTS_DIR}
ExecStart=/usr/bin/python3 -m http.server 8092 --bind 0.0.0.0
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable safe-alerts-http
systemctl restart safe-alerts-http

echo "== [5/6] udev auto start/stop on camera plug =="
cp ${PROJECT_DIR}/deploy/99-safecam.rules /etc/udev/rules.d/99-safecam.rules
udevadm control --reload-rules
# 关键：reload 只重新加载规则；要给"已经插着"的设备重跑，必须 trigger，
#       而且必须带 --action=add —— 规则匹配的是 ACTION=="add"，
#       而 udevadm trigger 默认是 --action=change：不带它不但建不出别名，
#       还会把已有的同类别名当过期链接删掉。
udevadm trigger --action=add --subsystem-match=video4linux || true
sleep 1
if [ -e /dev/safecam ]; then
  echo "udev rule installed, /dev/safecam -> $(readlink -f /dev/safecam)"
else
  echo "WARN: /dev/safecam 仍未生成。多半不是 USB(UVC) 摄像头，或规则没命中："
  echo "      查设备节点: v4l2-ctl --list-devices   命中检查: udevadm test /sys/class/video4linux/video9 2>&1 | grep -E \"safecam|ID_V4L_CAPABILITIES|ID_USB_DRIVER\""
  echo "      也可以直接把 safe-detect.service 的 --source 改成真实节点（如 /dev/video0）"
fi

echo "== [6/6] app units (not enabled by default) =="
cat > /etc/systemd/system/safe-detect.service <<EOF
[Unit]
Description=SafeDetect helmet/vest two-stage detection
After=network.target mosquitto.service
[Service]
Type=simple
User=firefly
Group=firefly
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 -u ${PROJECT_DIR}/app/safedetect_two_stage_rk3568.py --config ${PROJECT_DIR}/config/safe_config.json --source ${CAMERA_DEV} --noshow --alertdir ${PROJECT_DIR}/alerts --stream 8090
Restart=on-failure
RestartSec=5
SuccessExitStatus=3
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/safe-smoke.service <<EOF
[Unit]
Description=SafeDetect MQ-2 smoke sensor MQTT publisher
After=network.target mosquitto.service
[Service]
Type=simple
User=firefly
Group=firefly
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 -u ${PROJECT_DIR}/app/smoke_sensor.py --config ${PROJECT_DIR}/config/safe_config.json
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload

echo "=========================================="
echo " DONE. board IP: ${BOARD_IP}"
echo " MQTT:      ${BOARD_IP}:1883"
echo " Node-RED:  http://${BOARD_IP}:1880"
echo " ntfy:      http://${BOARD_IP}:8081"
echo " images:    http://${BOARD_IP}:8092/  (本包)"
echo " preview:   http://${BOARD_IP}:8090/  (本包)"
echo " phone:     ntfy App, server http://${BOARD_IP}:8081, topic safe_cam1 (着装+烟雾同一条)"
echo " optional:  sudo systemctl enable --now safe-detect safe-smoke"
echo "=========================================="