#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RK3568 安全帽/安全衣（PPE）检测入口（板端主程序，两级流水线）。

架构总览（数据流）：
  USB摄像头 
      1. 主循环：
             一级：yolov8n_headcut.rknn（COCO 80 类，只取 class0=person）-> 人体框
             二级：对每个人体裁剪 -> yolov8n_safe_headcut_i8.rknn（3 类）-> SAFE/PARTIAL/UNSAFE
      2. 合规状态机 SafetyMonitor（逐人 IoU 跟踪 + 连续帧确认 + 上升沿触发）
      3. 告警（MQTT -> Node-RED -> ntfy 手机推送 / 截图 + 前后录像）
      4. MJPEG 网页预览（stream_server，手机浏览器可远程看画面）

模型：
  一级 yolov8n_headcut.rknn          
  二级 yolov8n_safe_headcut_i8.rknn 

"""
import argparse
import csv
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite   # 板端 NPU 推理库（Rockchip）

# ---- 本地模块（同目录，职责单一） ----
from headcut_decode import (letterbox, scale_coords, decode_persons, decode_gear,
                            gear_status, filter_person_boxes, crop_person,
                            draw_gear, STATUS_SAFE, STATUS_UNSAFE, STATUS_PARTIAL)
from safety_rules import SafetyMonitor        # 着装违规状态机
from stream_server import (start_stream, stream_publish_loop,    # MJPEG 预览(8090)
                            get_lan_ip)
from config import load_config, pick_config    # 配置合并：默认 < JSON < 命令行
from mqtt_notify import Notifier              # MQTT 告警 + 心跳 + ntfy 链接
from clip_recorder import ClipRecorder, trim_old_records      # 告警前后录像
from rga_accel import RgaAccel                # RGA 硬件缩放（可选，失败自动退回 cv2）

# ---- 目录布局：本文件在 app/ 下 ----
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(APP_DIR)          # SafeDetect_V0.01/
DEFAULT_CONFIG = pick_config(PKG_ROOT)   # 私有 safe_config.json 优先，缺失则用公开模板
MODEL_DIRS = (os.path.join(PKG_ROOT, "models"), PKG_ROOT, APP_DIR)
# 视为"摄像头设备"的节点前缀：启动时按 config runtime.camera_nodes 覆盖
CAMERA_NODES = ["/dev/video", "/dev/safecam"]


def video_to_h264(path):
    """把 OpenCV 写出的视频转成 H.264（兼容性最好）。

    OpenCV 的 VideoWriter 只能用 mp4v(MJPG 等)编码，部分播放器/手机不支持，会报"加载视频文件时出错"。
    这里用 ffmpeg 转一道：优先板端硬编 h264_rkmpp，其次 libx264；ffmpeg 缺失或失败则保留原文件。
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return path
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        print("提示: 未装 ffmpeg，输出视频保持 mp4v 编码（个别播放器可能打不开）")
        return path
    base, ext = os.path.splitext(path)
    tmp = base + "_h264.mp4"
    for cmd in (
        ["ffmpeg", "-y", "-i", path, "-an", "-c:v", "h264_rkmpp", "-b:v", "1500k", tmp],
        ["ffmpeg", "-y", "-i", path, "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "23", tmp],
    ):
        try:
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=180)
            if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, path)
                print("已转成 H.264:", path)
                return path
        except Exception:
            continue
    print("转码失败，保留原编码（可用 ffmpeg 手动转）:", path)
    return path


def on_unsafe(frame, info, alert_path=None):
    """违规告警钩子：默认打印 + 存图。可在此扩展 GPIO / 语音 / HTTP 推送。"""
    print("!!! 检测到安全着装违规 !!!")
    print("   状态: %s | SAFE=%d PARTIAL=%d UNSAFE=%d"
          % (info["status_text"], info["safe"], info["partial"], info["unsafe"]))
    for p in info.get("alarm_persons", []):
        print("   人#%s %s 帽=%.2f 衣=%.2f"
              % (p.get("track_id"), p.get("status"),
                 float(p.get("helmet_conf", 0.0)), float(p.get("vest_conf", 0.0))))
    if alert_path:
        os.makedirs(os.path.dirname(alert_path), exist_ok=True)
        cv2.imwrite(alert_path, frame)
        print("   现场截图已保存:", alert_path)


def resolve_model(name, search_dirs=MODEL_DIRS):
    """模型名解析：绝对路径直接用；相对名依次在 models/ -> 包根 -> app/ 里找。"""
    if not name:
        return name
    if os.path.isabs(name):
        return name
    for d in search_dirs:
        cand = os.path.join(d, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(search_dirs[0], name)


def init_rknn(path, core_mask):
    """加载并初始化一个 RKNN 模型，失败抛 SystemExit（宁可直接退出也别跑出错误结果）。"""
    if not os.path.exists(path):
        raise SystemExit("找不到 RKNN 模型: " + path)
    net = RKNNLite(verbose=False)
    if net.load_rknn(path) != 0:
        raise SystemExit("load_rknn 失败: " + path)
    if net.init_runtime(core_mask=core_mask) != 0:
        raise SystemExit("init_runtime 失败: " + path)
    return net


def is_camera_source(source):
    """判断 source 是"摄像头设备"还是"视频文件/网络流"。

    - 数字(0/1) 或 /dev/videoX、udev 别名（如 /dev/safecam）-> 摄像头（V4L2 后端 + 分辨率预设）
    - rtsp:// / rtmp:// / http(s):// -> 网络流
    - 其它（含 .mp4/.avi/.mkv）-> 视频文件（默认后端直接打开，不做 V4L2 预设）

    前缀列表见 config runtime.camera_nodes（默认 /dev/video），换别名改配置即可。
    """
    s = str(source)
    if s.isdigit():
        return True
    return s.startswith(tuple(CAMERA_NODES))


def open_camera(source):
    """打开视频源。

    - 摄像头：V4L2 后端 + 低带宽格式预设（省带宽、降 CPU）
    - 视频文件/网络流：默认后端直接打开，不做分辨率/FOURCC 预设（那些是 V4L2 专有）
    """
    if not is_camera_source(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None, None
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0
        return cap, "VIDEO(%.1f fps)" % fps

    try:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    except Exception:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None, None
    mjpeg = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    picked_fmt = "YUYV"
    presets = [
        (mjpeg, 640, 480),
        (mjpeg, 1280, 720),
        (None, 640, 480),
        (None, 1280, 720),
        (None, 1920, 1080),
    ]
    for fourcc, w, h in presets:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) == w and
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) == h):
            picked_fmt = "MJPG" if fourcc else "YUYV"
            break
    return cap, picked_fmt


def describe_video_devices():
    """列出采集设备并给出建议节点，返回 (清单文本, 建议节点或 None)。

    v4l2-ctl 可用时用它（能区分 USB 摄像头 / MIPI 通道 / ISP 统计节点），
    否则退回 /dev/video* 列表。建议节点优先取设备名里带 usb/web/camera 的那一组的第一个节点
    —— USB 摄像头一般是 "capture" 节点，紧随其后的那个是 metadata 节点，不能用。
    """
    text = ""
    try:
        import subprocess
        r = subprocess.run(["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=5)
        if r.returncode == 0:
            text = r.stdout.decode(errors="replace").strip()
    except Exception:
        text = ""
    if not text:
        import glob
        devs = sorted(glob.glob("/dev/video*"))
        text = " ".join(devs) if devs else "（没找到 /dev/video*）"

    suggestion = None
    blocks = [b for b in text.split("\n\n") if b.strip()]
    for b in blocks:
        head = b.splitlines()[0].lower()
        if "usb" in head or "web" in head or "camera" in head:
            for ln in b.splitlines()[1:]:
                ln = ln.strip()
                if ln.startswith("/dev/video"):
                    suggestion = ln
                    break
        if suggestion:
            break
    if suggestion is None:
        for b in blocks:
            for ln in b.splitlines():
                ln = ln.strip()
                if ln.startswith("/dev/video"):
                    suggestion = ln
                    break
            if suggestion:
                break
    return text, suggestion


def open_camera_retry(source, stop, keep_alive, camera_failed=None, reconnected=None):
    """打开摄像头；打不开时 keep_alive=True 每 2 秒重试（直到 stop），返回 (cap, fmt)。"""
    cap, fmt = open_camera(source)
    recovery = False
    while cap is None:
        print("ERROR: 打不开视频源 %s" % source)
        print("       排查：1) systemd 单元/命令行里的 --source 是否就是这颗摄像头的节点")
        print("             2) 同一颗摄像头只能被一个进程独占（别的程序占着就打不开）")
        print("             3) 用 udev 别名（如 /dev/safecam）时先确认别名已生成，否则直接填 /dev/videoN")
        _txt, _sug = describe_video_devices()
        print("       当前采集设备（v4l2-ctl --list-devices）:")
        for _ln in _txt.splitlines():
            print("         " + _ln)
        if _sug:
            print("       建议把 --source 改成: %s" % _sug)
        if camera_failed is not None:
            camera_failed.set()
        if not keep_alive:
            return None, None
        recovery = True
        for _ in range(20):          # 2 秒一轮，期间可被 stop 打断
            if stop.is_set():
                return None, None
            time.sleep(2)
        cap, fmt = open_camera(source)
    if recovery:
        if camera_failed is not None:
            camera_failed.clear()
        if reconnected is not None:
            reconnected.set()
    return cap, fmt


def capture_loop(source, cap_q, stop, camera_failed=None, reconnected=None, keep_alive=False):
    """采集线程：断线自动重连；重连成功用 reconnected 事件通知主循环。
    keep_alive=True 时没有摄像头也不退出——主循环用占位画面继续提供远程预览。"""
    cap, fmt = open_camera_retry(source, stop, keep_alive, camera_failed, reconnected)
    if cap is None:
        if camera_failed is not None:
            camera_failed.set()
        stop.set()
        return
    print("视频源已打开:", source, "分辨率: %dx%d, FPS: %.1f, 格式: %s" % (
          cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
          cap.get(cv2.CAP_PROP_FPS), fmt))

    # 视频文件要按自身帧率喂帧：解码远快于推理，不限速会"快进"着几秒播完
    file_fps = 0.0
    if not is_camera_source(source):
        try:
            file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            file_fps = 0.0
        if not (1.0 <= file_fps <= 121.0):
            file_fps = 0.0
    next_t = time.time()

    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            if is_camera_source(source):
                print("摄像头读取失败，2 秒后重连...")
            else:
                print("视频播放结束，2 秒后从头重播（Ctrl+C 退出）")
            cap.release()
            time.sleep(2)
            cap, fmt = open_camera_retry(source, stop, keep_alive, camera_failed, reconnected)
            if cap is None:
                stop.set()
                return
            next_t = time.time()
            continue
        if cap_q.full():
            try:
                cap_q.get_nowait()
            except queue.Empty:
                pass
        cap_q.put(frame)
        if file_fps:                       # 视频文件：按原帧率节流，保证实时播放
            next_t += 1.0 / file_fps
            wait = next_t - time.time()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.time()
    if cap is not None:
        cap.release()
    stop.set()


def placeholder_frame(source, note=""):
    """没有摄像头时给远程预览用的占位画面。
    cv2 的 Hershey 字体不支持中文，所以画面用英文，并把排查命令写在画面上。"""
    w, h = 640, 480
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (0, 0, 200), 3)
    cv2.putText(img, "CAMERA OFFLINE", (108, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    cv2.putText(img, "source: %s" % source, (60, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(img, "check: ls /dev/video*", (60, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(img, "non-USB camera needs real device node", (60, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
    if note:
        cv2.putText(img, note, (60, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)
    return img

_THERMAL_CACHE = {"t": 0.0, "zones": []}


def read_temps(cache_sec=2.0):
    """读所有 thermal zone 温度(℃)，返回 [(标签, 温度), ...]；每 cache_sec 秒读一次。"""
    now = time.time()
    if now - _THERMAL_CACHE["t"] < cache_sec:
        return _THERMAL_CACHE["zones"]
    zones = []
    try:
        for i in range(16):
            tz = "/sys/class/thermal/thermal_zone%d" % i
            if not os.path.isdir(tz):
                break
            with open(os.path.join(tz, "type")) as f:
                typ = f.read().strip().lower()
            with open(os.path.join(tz, "temp")) as f:
                val = int(f.read().strip()) / 1000.0
            if "cpu" in typ:
                label = "CPU"
            elif "npu" in typ:
                label = "NPU"
            elif "gpu" in typ:
                label = "GPU"
            elif "soc" in typ:
                label = "SoC"
            else:
                label = "Zone%d" % i
            zones.append((label, val))
    except Exception:
        pass
    if not zones:  # 兜底：连 type 都读不到时直接取 zone0
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                zones.append(("SoC", int(f.read().strip()) / 1000.0))
        except Exception:
            pass
    zones.sort(key=lambda z: 0 if z[0] == "CPU" else 1)
    _THERMAL_CACHE.update({"t": now, "zones": zones})
    return zones


def preprocess_frame(frame, preproc, img_size):
    """letterbox + BGR->RGB；RGA 可用时硬件缩放，失败自动退回 OpenCV。"""
    if preproc is not None and preproc.enabled:
        res = preproc.letterbox(frame, (img_size, img_size))
        if res is not None:
            return res
    img, ratio, pad = letterbox(frame, (img_size, img_size))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ratio, pad


def classify_person(frame, box, gear_net, preproc, gear_img, gear_nc,
                    conf_gear, iou_gear, min_gear_conf, crop_pad):
    """二级：裁出这个人 -> letterbox -> 安全帽/安全衣推理 -> 三态结论。"""
    crop, _offset = crop_person(frame, box, crop_pad)
    if crop is None:
        return None
    rgb, _r, _p = preprocess_frame(crop, preproc, gear_img)
    outs = gear_net.inference(inputs=[rgb[None, ...]])
    gb, gs, gc = decode_gear(outs, conf=conf_gear, iou=iou_gear,
                             img_size=gear_img, nc=gear_nc)
    return gear_status(gb, gs, gc, min_conf=min_gear_conf)


def local_ipv4_list():
    """列出本机可用 IPv4（优先 `hostname -I`），用于打印浏览器/手机能直接打开的地址。"""
    ips = []
    try:
        import subprocess
        out = subprocess.check_output(["hostname", "-I"], timeout=3,
                                      stderr=subprocess.DEVNULL)
        for tok in out.decode(errors="replace").split():
            if tok and not tok.startswith("127.") and ":" not in tok and tok not in ips:
                ips.append(tok)
    except Exception:
        pass
    if not ips:
        try:
            ips.append(get_lan_ip())
        except Exception:
            pass
    return ips


def print_accessible_urls(port, notifier, cfg):
    """启动时把"能直接照抄"的地址打全：预览、告警图片、ntfy 服务器与订阅主题。"""
    ports = cfg.get("ports") or {}
    ntfy_port = int(ports.get("ntfy", 8081))
    alert_port = int(ports.get("alert_http", 8092))
    ips = []
    if notifier is not None and notifier.ts_ip:
        ips.append(notifier.ts_ip)
    for ip in local_ipv4_list():
        if ip not in ips:
            ips.append(ip)
    if not ips:
        ips = ["127.0.0.1"]
    prefix = (cfg.get("mqtt") or {}).get("topic_prefix", "safe")
    cam = cfg.get("camera_id", "cam1")
    print("---- 访问地址（局域网/手机直接打开）----")
    for ip in ips:
        print("  实时预览   http://%s:%d/" % (ip, port))
    print("  告警图片   http://%s:%d/" % (ips[0], alert_port))
    print("  手机 ntfy  服务器 http://%s:%d/   订阅主题 %s_%s（着装+烟雾统一通道）"
          % (ips[0], ntfy_port, prefix, cam))
    print("  （ntfy/告警图片端口见 config ports 节，由 deploy/install_notify_stack.sh 创建）")

    # 探一下告警图片服务在不在：ntfy 的附件是它自己去 127.0.0.1:<port> 拉的，
    # 服务不在就会表现为「看不到图片 / 下载附件失败」。
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:%d/" % alert_port, timeout=2).close()
    except Exception:
        print("  ⚠ 告警图片服务 %d 没响应：ntfy 会「看不到图片 / 下载附件失败」。" % alert_port)
        print("     补装：sudo bash deploy/install_alerts_http.sh")
        print("     端口被别的服务占用时，改 config ports.alert_http 后重装即可。")


def run_two_stage(frame, person_net, gear_net, preproc, cfg):
    """对一帧跑完整两级流水线，返回 (persons, timings)。persons 是每人的三态结论。"""
    md = cfg["model"]
    dc = cfg["detect"]
    person_img = int(md["person_img_size"])
    gear_img = int(md["gear_img_size"])

    t0 = time.time()
    rgb1, ratio1, pad1 = preprocess_frame(frame, preproc, person_img)
    t1 = time.time()
    outs1 = person_net.inference(inputs=[rgb1[None, ...]])
    t2 = time.time()
    pboxes, pscores, _pcls = decode_persons(
        outs1, conf=float(dc["conf_person"]), iou=float(dc["iou_person"]),
        img_size=person_img, nc=int(md["person_nc"]),
        person_class=int(md["person_class"]))
    pboxes = scale_coords(pboxes, ratio1, pad1, frame.shape[:2])
    pboxes, pscores = filter_person_boxes(
        pboxes, pscores, frame.shape[:2],
        min_area=float(dc["min_person_area"]),
        max_persons=int(dc["max_persons"]))
    t3 = time.time()

    # 演示用：把所有人的判定强制成同一状态（颜色、状态机、告警都跟着走）
    force_status = str(dc.get("force_status") or "").strip().upper()

    persons = []
    for box, score in zip(pboxes, pscores):
        st = classify_person(frame, box, gear_net, preproc, gear_img,
                             int(md["gear_nc"]), float(dc["conf_gear"]),
                             float(dc["iou_gear"]), float(dc["min_gear_conf"]),
                             float(dc["crop_pad"]))
        if st is None:
            continue
        if force_status in ("SAFE", "PARTIAL", "UNSAFE"):
            st = dict(st, status=force_status)
        persons.append({"box": [float(v) for v in box],
                        "score": float(score), **st})
    t4 = time.time()
    timings = {"pre": (t1 - t0) * 1000, "infer1": (t2 - t1) * 1000,
               "decode1": (t3 - t2) * 1000, "gear": (t4 - t3) * 1000,
               "total": (t4 - t0) * 1000}
    return persons, timings


def draw_overlay(frame, info, infer_ms, status_ok_color=(0, 255, 0)):
    """画状态行 + 温度 + 耗时。"""
    status = info["status_text"]
    color = (0, 0, 255) if info["alarm"] else status_ok_color
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    counts = info["counts"]
    cv2.putText(frame, "SAFE %d  PARTIAL %d  UNSAFE %d   %.0f ms"
                % (counts[STATUS_SAFE], counts[STATUS_PARTIAL], counts[STATUS_UNSAFE],
                   infer_ms),
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    temp_zones = read_temps()
    if temp_zones:
        temp_txt = "  ".join("%s %.1f\u00b0C" % (lab, val) for lab, val in temp_zones[:3])
        (tw, _), _ = cv2.getTextSize(temp_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame, temp_txt, (frame.shape[1] - tw - 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def main():
    base_dir = APP_DIR

    # ---------- 1) 命令行参数 ----------
    # 与配置相关的参数默认值都是 None：不传 = 用 safe_config.json / 默认值
    parser = argparse.ArgumentParser(description="RK3568 安全帽/安全衣(PPE)两级检测")
    parser.add_argument("--person-rknn", default=None,
                        help="一级人体检测 rknn（默认取 safe_config.json model.person_rknn）")
    parser.add_argument("--gear-rknn", default=None,
                        help="二级安全帽/安全衣 rknn（默认取 safe_config.json model.gear_rknn）")
    parser.add_argument("--img", default=None, help="单张图片路径（图片模式）")
    parser.add_argument("--out-video", default=None,
                        help="视频模式：把标注后的画面另存为视频（如 out.mp4），便于回放评估")
    parser.add_argument("--out", default="result_safe.jpg", help="图片模式输出路径")
    parser.add_argument("--dir", default=None,
                        help="批量目录模式：处理目录下所有 jpg，逐张出图 + 写 CSV 报告")
    parser.add_argument("--out-dir", default=None,
                        help="批量模式输出目录（默认 <alertdir>/batch_out）")
    parser.add_argument("--source", default=None,
                        help="视频源，默认取 safe_config.json runtime.source（0=USB, rtsp://..., 视频文件）")
    parser.add_argument("--conf-person", type=float, default=None,
                        help="一级人体置信度阈值（默认 detect.conf_person）")
    parser.add_argument("--conf-gear", type=float, default=None,
                        help="二级安全帽/安全衣置信度阈值（默认 detect.conf_gear）")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS iou 阈值（同时覆盖一级/二级，默认 detect.iou_*）")
    parser.add_argument("--force-status", default=None,
                        choices=["SAFE", "PARTIAL", "UNSAFE"],
                        help="演示用：把所有人体框的判定强制为该状态（默认不强制）")
    parser.add_argument("--max-persons", type=int, default=None,
                        help="每帧最多对几个最大的人跑二级（限制耗时，默认 detect.max_persons）")
    parser.add_argument("--bench", type=int, default=0,
                        help="性能基准：跑 N 帧，分阶段计时 pre/infer1/decode1/二级（不接摄像头也可跑）")
    parser.add_argument("--bench-img", default=None, help="基准用图片（默认找本目录 bus.jpg）")
    parser.add_argument("--core", default=None, choices=["AUTO", "0", "1", "2"],
                        help="NPU 核心，默认取 config runtime.core")
    parser.add_argument("--preprocess", default=None,
                        choices=["auto", "rga", "cv2"],
                        help="预处理: auto=RGA优先自动降级, rga=强制RGA, cv2=纯OpenCV")
    parser.add_argument("--fps", type=float, default=None,
                        help="估算帧率（预留给速度类规则），默认取 config runtime.fps")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--noshow", action="store_true")
    parser.add_argument("--alertdir", default=None,
                        help="告警截图/录像保存目录，默认取 config runtime.alertdir")
    parser.add_argument("--stream-width", type=int, default=None,
                        help="预览画面宽度（默认取 config runtime.stream_width，480；越小越省带宽）")
    parser.add_argument("--stream-quality", type=int, default=None,
                        help="预览 JPEG 质量 1-100（默认取 config runtime.stream_quality，60）")
    parser.add_argument("--stream", type=int, default=None,
                        help="HTTP 实时预览端口，默认取 config runtime.stream（本项目用 8090）")
    parser.add_argument("--mqtt-broker", default=None,
                        help="MQTT broker 地址（本机填 127.0.0.1，也可指向别的 mosquitto）")
    parser.add_argument("--camera-id", default=None,
                        help="MQTT 主题里用的摄像头编号，如 cam1")
    parser.add_argument("--ts-ip", default=None,
                        help="Tailscale 虚拟 IP（手动指定后不再自动探测，形如 100.x.y.z）")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="配置文件路径（私有 config/safe_config.json 优先，缺失用 safe_config_git.json）")

    args = parser.parse_args()
    if args.noshow:
        args.show = False

    core_map = {"AUTO": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0,
                "1": RKNNLite.NPU_CORE_1, "2": RKNNLite.NPU_CORE_2}

    # ---------- 2) 配置合并：命令行 > safe_config.json > config.py 默认 ----------
    overrides = {}
    if args.mqtt_broker is not None:
        overrides.setdefault("mqtt", {})["broker"] = args.mqtt_broker
    if args.camera_id is not None:
        overrides["camera_id"] = args.camera_id
    if args.ts_ip is not None:
        overrides.setdefault("notify", {})["ts_ip"] = args.ts_ip
    if args.person_rknn is not None:
        overrides.setdefault("model", {})["person_rknn"] = args.person_rknn
    if args.gear_rknn is not None:
        overrides.setdefault("model", {})["gear_rknn"] = args.gear_rknn
    if args.preprocess is not None:
        overrides["preprocess"] = args.preprocess
    if args.conf_person is not None:
        overrides.setdefault("detect", {})["conf_person"] = args.conf_person
    if args.conf_gear is not None:
        overrides.setdefault("detect", {})["conf_gear"] = args.conf_gear
    if args.iou is not None:
        overrides.setdefault("detect", {})["iou_person"] = args.iou
        overrides.setdefault("detect", {})["iou_gear"] = args.iou
    if args.force_status is not None:
        overrides.setdefault("detect", {})["force_status"] = args.force_status
    if args.max_persons is not None:
        overrides.setdefault("detect", {})["max_persons"] = args.max_persons
    for _arg, _key in ((args.source, "source"), (args.fps, "fps"),
                       (args.alertdir, "alertdir"), (args.stream, "stream"),
                       (args.stream_width, "stream_width"),
                       (args.stream_quality, "stream_quality"),
                       (args.core, "core")):
        if _arg is not None:
            overrides.setdefault("runtime", {})[_key] = _arg

    cfg = load_config(args.config, overrides)

    md = cfg["model"]
    dc = cfg["detect"]
    sr = cfg["safety_rules"]
    rt = cfg["runtime"]
    global CAMERA_NODES                      # 摄像头节点前缀按配置走（udev 别名换名不用改代码）
    CAMERA_NODES = list(rt.get("camera_nodes") or CAMERA_NODES)
    person_path = resolve_model(str(md["person_rknn"]))
    gear_path = resolve_model(str(md["gear_rknn"]))
    alertdir = rt["alertdir"]
    if not os.path.isabs(alertdir):          # 相对路径按包根解析，和 systemd 的 WorkingDirectory 解耦
        alertdir = os.path.join(PKG_ROOT, alertdir)
    stream = int(rt["stream"])
    stream_width = int(rt.get("stream_width", 480))
    stream_quality = int(rt.get("stream_quality", 60))
    core = rt["core"]

    # ---------- 3) 预处理 + 两个 NPU 模型 ----------
    preproc = RgaAccel(str(cfg["preprocess"]))
    print("预处理:", preproc.info())
    print("一级(人体)模型:", person_path, "img", md["person_img_size"],
          "nc", md["person_nc"], "class", md["person_class"])
    print("二级(安全帽/安全衣)模型:", gear_path, "img", md["gear_img_size"],
          "nc", md["gear_nc"])
    person_net = init_rknn(person_path, core_map[core])
    gear_net = init_rknn(gear_path, core_map[core])

    # ---------------- 性能基准（--bench N） ----------------
    if args.bench:
        n = max(3, int(args.bench))
        bench_img = None
        cands = [args.bench_img] if args.bench_img else []
        cands += [os.path.join(PKG_ROOT, "bus.jpg"), os.path.join(base_dir, "bus.jpg"), "bus.jpg"]
        for cand in cands:
            if cand and os.path.exists(cand):
                bench_img = cv2.imread(cand)
                if bench_img is not None:
                    print("基准: 使用", cand)
                    break
        if bench_img is None:
            bench_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print("基准: 未找到测试图，使用随机噪声帧（decode 耗时仅参考）")
        st = {"pre": 0.0, "infer1": 0.0, "decode1": 0.0, "gear": 0.0}
        for i in range(n + 3):
            persons, tm = run_two_stage(bench_img, person_net, gear_net, preproc, cfg)
            if i >= 3:
                for k in st:
                    st[k] += tm[k]
        print("---- 基准（%d 次平均，一级 %d / 二级 %d，每人 %.1f ms）----"
              % (n, md["person_img_size"], md["gear_img_size"], st["gear"] / max(n, 1)))
        tot = 0.0
        for k in ("pre", "infer1", "decode1", "gear"):
            avg = st[k] / n
            tot += avg
            print("  %-8s %8.2f ms" % (k, avg))
        print("  合计 %.2f ms/帧 (约 %.1f FPS)" % (tot, 1000.0 / max(tot, 1e-6)))
        person_net.release()
        gear_net.release()
        return 0

    # ---------- 4) 业务组件：状态机 / MQTT / 录像 / 推流 ----------
    monitor = SafetyMonitor(
        confirm_frames=int(sr["confirm_frames"]),
        clear_frames=int(sr["clear_frames"]),
        alert_on=tuple(sr.get("alert_on") or (STATUS_UNSAFE,)),
        track_iou=float(sr["track_iou"]),
        max_miss_frames=int(sr["max_miss_frames"]),
    )
    notifier = Notifier(cfg)
    recorder = ClipRecorder(cfg, alertdir)
    trim_old_records(alertdir, recorder.max_records)
    stop_hb = threading.Event()
    if notifier.enabled:
        threading.Thread(target=notifier.heartbeat_loop,
                         args=(stop_hb,), daemon=True).start()

    stream_q = None
    if stream:
        start_stream(stream, (cfg.get("network") or {}).get("lan_probe"))
        stream_q = queue.Queue(maxsize=1)
        threading.Thread(target=stream_publish_loop,
                         args=(stream_q, stream_width, stream_quality),
                         daemon=True).start()

    # ---------- 5) 图片模式（--img 单张调试） ----------
    if args.img:
        frame = cv2.imread(args.img)
        if frame is None:
            print("ERROR: 无法读取图片", args.img)
            return 2
        persons, tm = run_two_stage(frame, person_net, gear_net, preproc, cfg)
        # 单张图自检：确认帧数压到 1，让状态直接反映这张图（序列模式才用连续帧确认）
        single = SafetyMonitor(confirm_frames=1, clear_frames=1,
                               alert_on=tuple(sr.get("alert_on") or (STATUS_UNSAFE,)))
        info = single.update(persons, frame.shape[:2])
        for p in persons:
            draw_gear(frame, p["box"], p["status"], p["helmet_conf"],
                      p["vest_conf"], p.get("track_id"))
        draw_overlay(frame, info, tm["total"])
        print("耗时 pre %.1f / 一级 %.1f / 解码 %.1f / 二级 %.1f ms，检测到 %d 人"
              % (tm["pre"], tm["infer1"], tm["decode1"], tm["gear"], len(persons)))
        for p in persons:
            print("  人#%s %s 帽=%.2f 衣=%.2f 无衣=%.2f (二级框%d)"
                  % (p.get("track_id"), p["status"], p["helmet_conf"],
                     p["vest_conf"], p["no_vest_conf"], p["n_boxes"]))
        print("状态: %s" % info["status_text"])
        cv2.imwrite(args.out, frame)
        print("结果已保存:", args.out)
        if info["alarm"]:
            alert_path = os.path.join(alertdir, "safe_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
            on_unsafe(frame, info, alert_path)
            notifier.on_unsafe(frame, info, alert_path, None, tm["total"],
                               alert_on=cfg["safety_rules"].get("alert_on"))
        person_net.release()
        gear_net.release()
        return 0

    # ---------- 6) 批量目录模式（--dir，逐张出图 + CSV，和 rknn_two_stage_test.py 对齐） ----------
    if args.dir:
        out_dir = args.out_dir or os.path.join(alertdir, "batch_out")
        os.makedirs(out_dir, exist_ok=True)
        images = sorted(str(p) for p in __import__("pathlib").Path(args.dir).glob("*.jpg"))
        if not images:
            print("ERROR: 目录里没有 jpg:", args.dir)
            return 2
        report = os.path.join(out_dir, "safe_two_stage_result.csv")
        with open(report, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["image", "person", "x1", "y1", "x2", "y2",
                        "helmet_conf", "vest_conf", "no_vest_conf", "status"])
            for img_path in images:
                frame = cv2.imread(img_path)
                if frame is None:
                    print("无法读取", img_path)
                    continue
                persons, tm = run_two_stage(frame, person_net, gear_net, preproc, cfg)
                # 批量目录每张图是独立场景：各起一个状态机，人编号从 1 开始，不跨图累计
                per_img = SafetyMonitor(confirm_frames=1, clear_frames=1,
                                        alert_on=tuple(sr.get("alert_on") or (STATUS_UNSAFE,)))
                info = per_img.update(persons, frame.shape[:2])
                for pi, p in enumerate(persons):
                    x1, y1, x2, y2 = [int(v) for v in p["box"]]
                    draw_gear(frame, p["box"], p["status"], p["helmet_conf"],
                              p["vest_conf"], p.get("track_id"))
                    w.writerow([os.path.basename(img_path), pi, x1, y1, x2, y2,
                                round(p["helmet_conf"], 4), round(p["vest_conf"], 4),
                                round(p["no_vest_conf"], 4), p["status"]])
                draw_overlay(frame, info, tm["total"])
                out_img = os.path.join(out_dir, "two_" + os.path.basename(img_path))
                cv2.imwrite(out_img, frame)
                print("%s 人数=%d 状态=%s" % (os.path.basename(img_path),
                                              len(persons), info["status_text"]))
        print("报告:", report)
        person_net.release()
        gear_net.release()
        return 0

    # ---------- 7) 视频模式：采集线程 + 主循环 ----------
    source = rt["source"]
    source = int(source) if str(source).isdigit() else source
    keep_alive = bool(rt.get("keep_alive_no_camera", True))
    cap_q = queue.Queue(maxsize=2)
    stop = threading.Event()
    camera_failed = threading.Event()
    reconnected = threading.Event()
    threading.Thread(target=capture_loop,
                     args=(source, cap_q, stop, camera_failed, reconnected, keep_alive),
                     daemon=True).start()

    print("安全着装检测启动，Ctrl+C 退出（--noshow 时无窗口）...")
    frame_id = 0
    startup_notified = False
    camera_lost_notified = False
    alert_on = cfg["safety_rules"].get("alert_on")

    # 视频模式可选：把标注画面写成视频文件（懒创建，第一帧到达时才知道分辨率）
    out_video = args.out_video
    vw = None
    out_frames = 0
    out_fps = float(rt.get("fps") or 25.0)   # 输出视频帧率：优先取视频文件自身帧率，其次 runtime.fps
    if out_video:
        try:
            _c = cv2.VideoCapture(source)
            _f = float(_c.get(cv2.CAP_PROP_FPS) or 0.0)
            _c.release()
            if 1.0 <= _f <= 121.0:
                out_fps = _f
        except Exception:
            pass
        print("标注视频将输出到:", out_video, "(%.1f fps)" % out_fps)

    def push_stream(fr):
        """把画面塞进 MJPEG 预览队列（满了丢旧帧，不阻塞主循环）。"""
        if not stream:
            return
        if stream_q.full():
            try:
                stream_q.get_nowait()
            except queue.Empty:
                pass
        stream_q.put(fr)

    try:
        while not stop.is_set():
            # 摄像头恢复（从"无摄像头"或掉线中回来）：发一次"摄像头已连接"
            if reconnected.is_set():
                reconnected.clear()
                camera_lost_notified = False
                if notifier.enabled:
                    notifier.publish_status("reconnect")

            # 没有摄像头：服务不退，改用占位画面继续提供远程预览（插回来自动恢复检测）
            if cap_q.empty() and camera_failed.is_set():
                if not camera_lost_notified:
                    camera_lost_notified = True
                    if notifier.enabled:
                        notifier.publish_status("camera_lost")
                    print("提示: 摄像头不可用，预览已切换为占位画面；插上摄像头会自动恢复检测")
                frame = placeholder_frame(source)
                push_stream(frame)
                if frame_id % 25 == 0:
                    print("#%d: 摄像头未连接（预览占位中，source=%s）" % (frame_id, source))
                if args.show:
                    cv2.imshow("safe_detect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop.set()
                frame_id += 1
                time.sleep(0.15)
                continue

            if cap_q.empty():
                time.sleep(0.005)
                continue
            frame = cap_q.get()   # 取帧（队列空则让出 CPU，不阻塞采集）

            # 摄像头首帧到达：发一次"摄像头已连接"到 MQTT -> Node-RED -> ntfy
            if not startup_notified and notifier.enabled:
                startup_notified = True
                notifier.publish_status("startup")

            persons, tm = run_two_stage(frame, person_net, gear_net, preproc, cfg)
            info = monitor.update(persons, frame.shape[:2])

            for p in persons:
                draw_gear(frame, p["box"], p["status"], p["helmet_conf"],
                          p["vest_conf"], p.get("track_id"))
            draw_overlay(frame, info, tm["total"])

            # 告警上升沿：截图 + MQTT/ntfy + 录像
            if info["new_alert"]:
                ts = time.strftime("%Y%m%d_%H%M%S")
                alert_path = os.path.join(alertdir, "safe_%s.jpg" % ts)
                video_path = os.path.splitext(alert_path)[0] + ".mp4"
                on_unsafe(frame, info, alert_path)
                if notifier.on_unsafe(frame, info, alert_path, video_path,
                                      tm["total"], alert_on=alert_on):
                    recorder.start(alert_path)

            recorder.feed(frame)   # 告警激活时自动缓存前后片段

            push_stream(frame)

            if out_video:
                if vw is None:
                    _h, _w = frame.shape[:2]
                    _fourcc = cv2.VideoWriter_fourcc(
                        *("mp4v" if out_video.lower().endswith(".mp4") else "MJPG"))
                    vw = cv2.VideoWriter(out_video, _fourcc, out_fps, (_w, _h))
                    if not vw.isOpened():
                        print("WARN: 无法创建输出视频 %s（检查路径/扩展名）" % out_video)
                        vw = None
                        out_video = None
                if vw is not None:
                    vw.write(frame)
                    out_frames += 1

            if frame_id % 10 == 0:
                print("#%d: %s, pre %.1f/一级 %.1f/解码 %.1f/二级 %.1f ms, %d 人"
                      % (frame_id, info["status_text"], tm["pre"], tm["infer1"],
                         tm["decode1"], tm["gear"], len(persons)))

            if args.show:
                cv2.imshow("safe_detect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
            frame_id += 1
    except KeyboardInterrupt:
        pass

    if vw is not None:
        vw.release()
        print("标注视频已保存: %s (%d 帧, %.1f fps)" % (out_video, out_frames, out_fps))
        video_to_h264(out_video)

    stop_hb.set()
    person_net.release()
    gear_net.release()
    if args.show:
        cv2.destroyAllWindows()
    if camera_failed.is_set() and not keep_alive:
        return 3
    return 0

if __name__ == "__main__":
    sys.exit(main())
