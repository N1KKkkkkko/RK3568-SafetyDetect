# -*- coding: utf-8 -*-
"""MQTT 通知器：安全着装告警 + 心跳。

主题约定：<topic_prefix>/<camera_id>/alert | heartbeat | status
  - alert:     违规告警 JSON（含截图/视频路径、每个人体框的三态结论、infer 耗时）
  - heartbeat: 每 heartbeat_interval 秒发布 retain 心跳，用于监控设备在线
  - status:    摄像头启动/重连提示（Node-RED 转成"摄像头已连接"通知）
告警带 cooldown 秒冷却：同一事件短时间内不重复推送（去重）。
broker 为空时自动禁用（纯检测模式），不影响主流程。
"""
import json
import os
import socket
import subprocess
import time


def _get_ts_ip(probes=None):
    """取"手机端能打开的"地址：优先 Tailscale 虚拟 IP(100.x)，其次局域网 IP，最后 127.0.0.1。

    这个地址会随告警消息发给 Node-RED，成为通知里截图/录像/实时画面的主机名，
    所以顺序不能乱：有 Tailscale 就用 100.x（离家也能看），没有就用本机局域网 IP。
    probes 是探测目标列表（config network.lan_probe），换现场网络不用改代码。
    """
    # 1) Tailscale 虚拟 IP
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], timeout=5,
                                      stderr=subprocess.DEVNULL)
        ip = out.decode().strip().splitlines()[0].strip()
        if ip.startswith("100."):
            return ip
    except Exception:
        pass
    # 2) hostname -I 的第一个非回环 IPv4（板端最稳，不依赖能不能连通某个外网地址）
    try:
        out = subprocess.check_output(["hostname", "-I"], timeout=3,
                                      stderr=subprocess.DEVNULL)
        for tok in out.decode(errors="replace").split():
            if tok and not tok.startswith("127.") and ":" not in tok:
                return tok
    except Exception:
        pass
    # 3) UDP connect 技巧：只挑路由、不发包
    for probe in (probes or ["1.1.1.1"]):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
    return "127.0.0.1"


class Notifier:
    """MQTT 发布器：安全着装告警 + 心跳 + 摄像头状态。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.camera_id = cfg.get("camera_id", "cam1")
        self.cooldown = float((cfg.get("alerts") or {}).get("cooldown", 60))
        self.last_alert_ts = 0.0
        self.client = None
        self.enabled = False
        # 通知链接主机：safe_config.json notify.ts_ip 手动指定优先，留空自动探测
        self.ts_ip = str((cfg.get("notify") or {}).get("ts_ip") or "").strip()
        if not self.ts_ip:
            self.ts_ip = _get_ts_ip((cfg.get("network") or {}).get("lan_probe"))
        self._connect()

    def _connect(self):
        # broker 未配置 -> 禁用 MQTT；paho 缺失/连接失败也只告警不崩溃
        mq = self.cfg.get("mqtt") or {}
        broker = str(mq.get("broker", "")).strip()
        if not broker:
            print("MQTT 未配置 broker，告警仅本地保存（不影响检测）")
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("WARN: 当前用户没装 paho-mqtt，MQTT 已禁用。")
            print("      服务以 root 跑时，`pip3 install --user` 装的包对它不可见：")
            print("      sudo apt install python3-paho-mqtt  或  sudo pip3 install paho-mqtt")
            return
        client = mqtt.Client(client_id="%s-%d" % (self.camera_id, os.getpid()))
        if mq.get("username"):
            client.username_pw_set(mq.get("username"), mq.get("password", ""))
        try:
            client.connect(broker, int(mq.get("port", 1883)), keepalive=60)
            client.loop_start()
        except Exception as e:
            print("WARN: MQTT 连接 %s:%s 失败: %s" % (broker, mq.get("port", 1883), e))
            return
        self.client = client
        self.enabled = True
        print("MQTT 已启用: broker=%s:%s topic=%s/%s/#"
              % (broker, mq.get("port", 1883),
                 mq.get("topic_prefix", "safe"), self.camera_id))

    def topic(self, kind):
        prefix = (self.cfg.get("mqtt") or {}).get("topic_prefix", "safe")
        return "%s/%s/%s" % (prefix, self.camera_id, kind)

    def publish(self, kind, payload, retain=False):
        """统一发布入口：topic = <prefix>/<camera_id>/<kind>"""
        if not self.client:
            return
        qos = int((self.cfg.get("mqtt") or {}).get("qos", 1))
        try:
            self.client.publish(self.topic(kind),
                                json.dumps(payload, ensure_ascii=False),
                                qos=qos, retain=retain)
        except Exception as e:
            print("WARN: MQTT 发布 %s 失败: %s" % (kind, e))

    def publish_status(self, event):
        """摄像头启动/重连提示（Node-RED 转成"摄像头已连接"+ 实时画面按钮）。"""
        self.publish("status", {
            "event": event,
            "camera_id": self.camera_id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ts_ip": self.ts_ip,
        })

    def heartbeat_loop(self, stop):
        """后台线程：周期发心跳，stop 事件用于退出。"""
        interval = float((self.cfg.get("mqtt") or {}).get("heartbeat_interval", 30))
        t0 = time.time()
        while not stop.is_set():
            self.publish("heartbeat", {
                "event": "heartbeat",
                "camera_id": self.camera_id,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "uptime": round(time.time() - t0, 1),
                "alive": True,
            }, retain=True)
            stop.wait(interval)

    def on_unsafe(self, frame, info, image_path, video_path=None, infer_ms=None,
                  alert_on=None):
        """着装违规上报。返回 True = 这是一条新告警（未被冷却抑制），主程序据此启动录像；
        冷却期内重复触发返回 False，避免同一个人连发通知。"""
        now = time.time()
        if now - self.last_alert_ts < self.cooldown:
            print("告警冷却中(%.0fs)，跳过推送" % self.cooldown)
            return False
        self.last_alert_ts = now

        persons = []
        for p in (info.get("alarm_persons") or info.get("persons") or []):
            persons.append({
                "track_id": p.get("track_id"),
                "status": p.get("status"),
                "box": [int(round(v)) for v in p.get("box", [])],
                "score": round(float(p.get("score", 0.0)), 3),
                "helmet_conf": round(float(p.get("helmet_conf", 0.0)), 3),
                "vest_conf": round(float(p.get("vest_conf", 0.0)), 3),
                "no_vest_conf": round(float(p.get("no_vest_conf", 0.0)), 3),
            })

        cfg_alert_on = (self.cfg.get("safety_rules") or {}).get("alert_on")
        payload = {
            "event": "unsafe",
            "camera_id": self.camera_id,
            "ts_ip": self.ts_ip,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": image_path,
            "video": video_path,
            "infer_ms": None if infer_ms is None else round(infer_ms, 1),
            "status_text": info.get("status_text"),
            "worst": info.get("worst"),
            "counts": info.get("counts"),
            "unsafe": info.get("unsafe"),
            "partial": info.get("partial"),
            "alert_on": list(alert_on or cfg_alert_on or []),
            "persons": persons,
        }
        if self.client:
            self.publish("alert", payload)
            print("MQTT 告警已发布:", self.topic("alert"))
        else:
            print("WARN: MQTT 未启用，告警未推送（仅本地保存截图/录像）")
        return True