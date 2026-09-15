# -*- coding: utf-8 -*-
"""ClipRecorder：告警前后视频片段（MJPG 原始录制 -> H.264 MP4）。

告警前后录像：滚动缓存前 N 秒 + 告警后录制 M 秒，文件以 safe_ 为前缀，
截图与录像用同一时间戳命名（safe_<时间>.jpg / .mp4），成对保存。
"""
import os
import time
from collections import deque

import cv2

# 告警文件名前缀：safe_<yyyymmdd_HHMMSS>.jpg / .mp4
PREFIX = "safe"


def trim_old_records(alertdir, max_records=100, prefix=PREFIX):
    """保留最近 max_records 组记录（每组 = 同名的时间戳图片 + 视频）。
    文件按基名（<prefix>_<ts>.*）分组，最早的分组整组删除。"""
    if not alertdir or not os.path.isdir(alertdir):
        return
    groups = {}
    for name in os.listdir(alertdir):
        if not name.startswith(prefix + "_"):
            continue
        base = name.split(".")[0]
        groups.setdefault(base, []).append(os.path.join(alertdir, name))
    bases = sorted(groups.keys())
    for base in (bases[:-max_records] if max_records > 0 else bases):
        for path in groups[base]:
            try:
                os.remove(path)
            except OSError:
                pass


class ClipRecorder:
    """滚动缓存前 N 秒 + 告警后录制 M 秒，保存为 MP4（不支持时退回 AVI）。"""

    def __init__(self, cfg, alertdir="alerts", max_w=640, prefix=PREFIX):
        al = cfg.get("alerts") or {}
        self.pre_n = max(1, int(float(al.get("video_pre_sec", 3.0)) * 8))
        self.post_n = max(1, int(float(al.get("video_post_sec", 5.0)) * 8))
        self.alertdir = alertdir
        self.max_records = int(al.get("max_records", 100))
        self.max_w = max_w
        self.prefix = prefix
        self.buf = deque(maxlen=self.pre_n)
        self.writer = None
        self.post_left = 0
        self.path = None

    def feed(self, frame):
        h, w = frame.shape[:2]
        if w > self.max_w:
            nh = int(round(h * self.max_w / w))
            frame = cv2.resize(frame, (self.max_w, nh), interpolation=cv2.INTER_AREA)
        if self.writer is not None:
            self.writer.write(frame)
            self.post_left -= 1
            if self.post_left <= 0:
                self.stop()
        else:
            self.buf.append(frame)

    def start(self, alert_path=None):
        if self.writer is not None:
            return self.path
        os.makedirs(self.alertdir, exist_ok=True)
        if alert_path:
            # 和告警截图共用同一个时间戳基名 -> safe_<ts>.jpg / safe_<ts>.mp4 成对
            base = os.path.splitext(os.path.basename(alert_path))[0]
        else:
            base = "%s_%s" % (self.prefix, time.strftime("%Y%m%d_%H%M%S"))
        h, w = self.buf[-1].shape[:2] if self.buf else (360, 640)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # 优先直接写 MP4（MJPG 编码）；OpenCV 不支持时退回 AVI，保证录像不会静默失败。
        path_mp4 = os.path.join(self.alertdir, base + ".mp4")
        writer = cv2.VideoWriter(path_mp4, fourcc, 8.0, (w, h))
        if writer.isOpened():
            self.path = path_mp4
        else:
            writer.release()
            path_avi = os.path.join(self.alertdir, base + ".avi")
            writer = cv2.VideoWriter(path_avi, fourcc, 8.0, (w, h))
            self.path = path_avi
        self.writer = writer
        for f in self.buf:
            self.writer.write(f)
        self.post_left = self.post_n
        return self.path

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._convert_mp4()
        trim_old_records(self.alertdir, self.max_records, self.prefix)

    def _convert_mp4(self):
        """把原始 MJPG 片段转成 H.264 MP4：优先板端硬编 h264_rkmpp，退回 libx264，
        两个都失败就保留原文件（不额外占空间，只是播放器兼容性差一点）。"""
        if not self.path or not os.path.exists(self.path):
            return
        tmp = os.path.splitext(self.path)[0] + "_h264.mp4"
        try:
            import subprocess
        except ImportError:
            print("告警视频已保存:", self.path)
            return
        cmds = [
            ["ffmpeg", "-y", "-i", self.path, "-an", "-c:v", "h264_rkmpp", "-b:v", "1200k", tmp],
            ["ffmpeg", "-y", "-i", self.path, "-an", "-c:v", "libx264", "-preset", "ultrafast",
             "-b:v", "1200k", tmp],
        ]
        for cmd in cmds:
            try:
                r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=90)
                if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    os.remove(self.path)
                    os.replace(tmp, self.path)
                    print("告警视频已保存:", self.path)
                    return
            except Exception:
                continue
        print("告警视频已保存（未转 H.264）:", self.path)