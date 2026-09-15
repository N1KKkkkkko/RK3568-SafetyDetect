# -*- coding: utf-8 -*-
"""
优先级：命令行 > safe_config.json > DEFAULTS 。
改参数只需改 命令行参数 / safe_config.json / DEFAULTS 三个中之一。
"""
import json
import os

# 唯一默认值来源（safe_config.json 里没有的键才用到这里）
DEFAULTS = {
    "camera_id": "cam1",
    # 网络探测：只为"猜本机 IP"用（UDP connect 只挑路由，不发包），按现场路由填即可
    "network": {"lan_probe": ["1.1.1.1"]},
    # 端口：本包与板子级服务端口；和别的项目同板共存时改这里（预览端口也可 --stream 临时覆盖）
    "ports": {"ntfy": 8081, "alert_http": 8092},
    "mqtt": {"broker": "", "port": 1883, "username": "", "password": "",
             "topic_prefix": "safe", "qos": 1, "heartbeat_interval": 30},
    "notify": {"ts_ip": ""},   # 通知链接主机：留空=自动探测 Tailscale 虚拟 IP
    "alerts": {"cooldown": 60, "video_pre_sec": 3.0, "video_post_sec": 5.0,
               "max_records": 100},
    "smoke": {"topic_prefix": "safe",
              "do": {"gpiochip": "gpiochip1", "line": 1, "active_low": True},
              "ao": {"enable": True, "alarm": False, "alarm_raw": 150,
                     "alarm_hysteresis": 10, "alarm_volt": None,
                     "iio_path": "/sys/bus/iio/devices/iio:device0/in_voltage0_raw",
                     "vref": 1.8, "bits": 10, "divider": 0.4},
              "poll_sec": 0.5, "debounce": 3, "cooldown": 120,
              "value_interval": 10, "log_interval": 10},
    # 安全着装判定：连续 confirm_frames 帧违规才告警；连续 clear_frames 帧合规才解除
    "safety_rules": {"confirm_frames": 5, "clear_frames": 15,
                     "alert_on": ["UNSAFE", "PARTIAL"],
                     "track_iou": 0.30, "max_miss_frames": 30},
    # detect_person 用一级模型，detect_gear 用二级模型
    "detect": {"conf_person": 0.35, "iou_person": 0.45,
               "conf_gear": 0.25, "iou_gear": 0.45,
               "crop_pad": 0.08, "max_persons": 5,
               "min_person_area": 0.0025, "min_gear_conf": 0.0, "force_status": ""},
    "model": {"person_rknn": "yolov8n_headcut.rknn", "person_img_size": 640,
              "person_nc": 80, "person_class": 0,
              "gear_rknn": "yolov8n_safe_headcut_i8.rknn", "gear_img_size": 640,
              "gear_nc": 3},
    "preprocess": "auto",
    # 运行时参数：默认来自这里，命令行 --xxx 覆盖
    # 端口约定：MJPEG 预览 8090、告警图片/录像 8092（避开 8080/8082 等常用端口）
    "runtime": {"source": "0", "fps": 20.0, "alertdir": "alerts",
                "stream": 8090,
                # 这些前缀当作摄像头节点（V4L2 后端）；用 udev 别名加进来即可，如 /dev/safecam
                "camera_nodes": ["/dev/video", "/dev/safecam"],
                "keep_alive_no_camera": True, "core": "AUTO"},
}

# 私有配置文件名 / 公开模板文件名（两者字段一致，只是取值不同）
PRIVATE_CONFIG = "safe_config.json"
PUBLIC_CONFIG = "safe_config_git.json"


def pick_config(pkg_root):
    """选配置文件：优先私有 config/safe_config.json，没有（如刚 clone）就用公开模板。

    safe_config.json 放真实部署值（本机 IP、传感器接线与阈值、broker 账号等），已在 .gitignore 里；
    safe_config_git.json 是可上传的默认模板，字段完全一致，缺哪个键就退回 DEFAULTS。
    """
    cfg_dir = os.path.join(pkg_root, "config")
    private = os.path.join(cfg_dir, PRIVATE_CONFIG)
    return private if os.path.exists(private) else os.path.join(cfg_dir, PUBLIC_CONFIG)


def _deep_update(base, extra):
    """递归合并：extra 中的 dict 键与 base 同名时逐层合并，否则覆盖。"""
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config(path, overrides=None):
    """返回合并后的配置：DEFAULTS < safe_config.json < overrides(命令行)。"""
    cfg = json.loads(json.dumps(DEFAULTS))  # 深拷贝，避免污染默认值
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _deep_update(cfg, data)
        except FileNotFoundError:
            pass
        except Exception as e:
            print("WARN: cannot parse config %s: %s" % (path, e))
    _deep_update(cfg, overrides)
    return cfg