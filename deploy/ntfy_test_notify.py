#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntfy 测试通知：标题带 虚拟IP:端口，点通知/按钮直接跳浏览器看实时画面。

用法（板端）:
  python3 deploy/ntfy_test_notify.py   # 在包根执行
可选:
  --topic safe_cam1   推送主题（默认取 safe_config.json 的 <prefix>_<camera_id>）
  --ntfy http://127.0.0.1:8081
  --port 8090         实时画面端口（默认 8090，即本包 MJPEG 预览）
ts_ip 来源（与主程序一致）：safe_config.json notify.ts_ip 手动指定 > tailscale ip -4 自动探测。
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # SafeDetect_V0.01/
sys.path.insert(0, os.path.join(BASE, "app"))
from config import load_config, pick_config
from mqtt_notify import _get_ts_ip


def main():
    cfg = load_config(pick_config(BASE))          # 端口/主题都按配置来，不写死
    ntfy_port = int((cfg.get("ports") or {}).get("ntfy", 8081))
    stream_port = int((cfg.get("runtime") or {}).get("stream", 8090))

    ap = argparse.ArgumentParser(description="ntfy 测试通知（点开直接看实时画面）")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--ntfy", default="http://127.0.0.1:%d" % ntfy_port)
    ap.add_argument("--port", type=int, default=stream_port)
    args = ap.parse_args()

    ts_ip = str((cfg.get("notify") or {}).get("ts_ip") or "").strip() or _get_ts_ip(
        (cfg.get("network") or {}).get("lan_probe"))
    topic = args.topic or "%s_%s" % (cfg["mqtt"]["topic_prefix"], cfg["camera_id"])
    stream_url = "http://%s:%d/" % (ts_ip, args.port)

    if ts_ip in ("127.0.0.1", "0.0.0.0", ""):
        print("警告: 当前主机=%s，手机可能打不开。请在 safe_config.json notify.ts_ip 手动填 Tailscale 虚拟 IP" % ts_ip)

    payload = json.dumps({
        "topic": topic,
        "title": "测试通知 %s:%d" % (ts_ip, args.port),   # 标题直接显示 IP:端口
        "message": "点击通知或下方按钮，直接打开浏览器查看实时画面",
        "priority": 3,
        "click": stream_url,                               # 点通知本体直接跳流
        "actions": [{
            "action": "view",
            "label": "查看实时画面",
            "url": stream_url
        }],
    })

    r = subprocess.run(
        ["curl", "-sS", "-m", "10",
         "-H", "Content-Type: application/json",
         "-d", payload, args.ntfy.rstrip("/")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 0:
        print("已发送测试通知: 主题=%s  标题=%s" % (topic, "测试通知 %s:%d" % (ts_ip, args.port)))
        print("点击跳转: %s" % stream_url)
    else:
        print("发送失败:", r.stderr.decode(errors="replace").strip())


if __name__ == "__main__":
    main()