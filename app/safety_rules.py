# -*- coding: utf-8 -*-
"""
SafetyMonitor：安全着装违规状态机（逐人 IoU 跟踪 + 连续帧确认）。

SAFE: 带了安全帽和安全衣
PARTIAL:带了安全帽或安全衣，但不是全套
UNSAFE: 都没带

输入：每帧二级模型给出的人体框与三态结论（SAFE / PARTIAL / UNSAFE）。
处理：
  1. 用 IoU 把当前帧人体框与上一帧的“轨迹”贪心匹配（>= track_iou），
     匹配不上的人当新轨迹，连续丢帧超过 max_miss_frames 的轨迹删除；
  2. 每条轨迹各自累计“连续违规帧数 confirm / 连续合规帧数 clear”：
     连续违规 >= confirm_frames -> 该轨迹进入告警；
     连续合规 >= clear_frames   -> 解除告警；
  3. 只有“从正常变成告警”的那一帧返回 new_alert=True（上升沿），
     避免同一个人在告警期内反复推送（配合 Notifier 的 cooldown 双保险）。

这样单帧误检不会推送，同一个人短暂丢帧/换框也不会清空连击。
"""
import numpy as np

STATUS_SAFE = "SAFE"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNSAFE = "UNSAFE"


def iou_xyxy(a, b):
    """两个 [x1,y1,x2,y2] 框的 IoU。"""
    xx1 = max(float(a[0]), float(b[0]))
    yy1 = max(float(a[1]), float(b[1]))
    xx2 = min(float(a[2]), float(b[2]))
    yy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / (aa + bb - inter + 1e-9)


class _Track:
    __slots__ = ("id", "box", "score", "status", "confirm", "clear", "miss", "alarm")

    def __init__(self, tid, box, score, status, violating):
        self.id = tid
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.status = status
        self.confirm = 1 if violating else 0
        self.clear = 0 if violating else 1
        self.miss = 0
        self.alarm = False


class SafetyMonitor:
    """安全帽/安全衣违规判定：逐人跟踪 + 连续帧确认 + 上升沿触发。"""

    def __init__(self, confirm_frames=5, clear_frames=15,
                 alert_on=("UNSAFE", "PARTIAL"), track_iou=0.30,
                 max_miss_frames=30):
        self.confirm_frames = max(1, int(confirm_frames))  # 连续几帧违规才告警
        self.clear_frames = max(1, int(clear_frames))      # 连续几帧合规才解除
        self.alert_on = tuple(alert_on or (STATUS_UNSAFE,))  # 哪些状态算违规
        self.track_iou = float(track_iou)                  # 同一人的框匹配门槛
        self.max_miss_frames = int(max_miss_frames)        # 丢帧多久删轨迹
        self.tracks = []
        self._next_id = 1
        self.last_info = {}

    # ---------------- 内部 ----------------
    def _violating(self, status):
        return status in self.alert_on

    def _advance(self, tr, status):
        """按新状态推进一条轨迹的计数，返回本次是否“刚进入告警”。"""
        tr.status = status
        tr.miss = 0
        if self._violating(status):
            tr.confirm += 1
            tr.clear = 0
        else:
            tr.clear += 1
            tr.confirm = 0
        flipped = False
        if tr.confirm >= self.confirm_frames and not tr.alarm:
            tr.alarm = True
            flipped = True
        if tr.clear >= self.clear_frames and tr.alarm:
            tr.alarm = False
        return flipped

    def _annotate(self, person, tr):
        person["track_id"] = tr.id
        person["alarm"] = tr.alarm
        person["confirm"] = tr.confirm
        person["violating"] = self._violating(tr.status)

    # ---------------- 主入口 ----------------
    def update(self, persons, frame_shape=None):
        """persons: [{"box":[x1,y1,x2,y2], "score":.., "status":..,
                      "helmet_conf":.., "vest_conf":..}, ...]
        就地给每个 person 补 track_id/alarm/confirm/violating 字段。"""
        persons = list(persons or [])
        old_tracks = self.tracks

        # 1) IoU 贪心匹配（iou 降序；同分按轨迹/人顺序，保证结果稳定）
        pairs = []
        for ti, tr in enumerate(old_tracks):
            for pi, p in enumerate(persons):
                v = iou_xyxy(tr.box, p["box"])
                if v >= self.track_iou:
                    pairs.append((v, -ti, -pi))
        pairs.sort(reverse=True)

        matched_track = [False] * len(old_tracks)
        matched_person = [False] * len(persons)
        new_alert = False
        for _v, nti, npi in pairs:
            ti, pi = -nti, -npi
            if matched_track[ti] or matched_person[pi]:
                continue
            matched_track[ti] = True
            matched_person[pi] = True
            tr = old_tracks[ti]
            tr.box = np.asarray(persons[pi]["box"], dtype=np.float32)
            tr.score = float(persons[pi].get("score", 0.0))
            if self._advance(tr, persons[pi]["status"]):
                new_alert = True
            self._annotate(persons[pi], tr)

        # 2) 存活轨迹：匹配上的保留；没匹配上的累计丢帧，超时才删除
        survivors = []
        for ti, tr in enumerate(old_tracks):
            if matched_track[ti]:
                survivors.append(tr)
                continue
            tr.miss += 1
            if tr.miss <= self.max_miss_frames:
                survivors.append(tr)

        # 3) 本帧新出现、没匹配上任何旧轨迹的人 -> 新建轨迹
        for pi, p in enumerate(persons):
            if matched_person[pi]:
                continue
            tr = _Track(self._next_id, p["box"], p.get("score", 0.0),
                        p["status"], self._violating(p["status"]))
            self._next_id += 1
            if tr.confirm >= self.confirm_frames:
                tr.alarm = True
                new_alert = True
            self._annotate(p, tr)
            survivors.append(tr)
        self.tracks = survivors

        # 4) 汇总
        counts = {STATUS_SAFE: 0, STATUS_PARTIAL: 0, STATUS_UNSAFE: 0}
        for p in persons:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        violations = sum(1 for p in persons if self._violating(p["status"]))
        alarm_persons = [p for p in persons if p.get("alarm")]
        any_alarm = any(tr.alarm for tr in self.tracks)

        if counts[STATUS_UNSAFE]:
            worst = STATUS_UNSAFE
        elif counts[STATUS_PARTIAL]:
            worst = STATUS_PARTIAL
        else:
            worst = STATUS_SAFE

        max_confirm = max([tr.confirm for tr in self.tracks], default=0)
        if not persons and not any_alarm:
            status_text = "无人"
        elif any_alarm:
            status_text = "违规告警 %d人" % max(1, len(alarm_persons))
        elif max_confirm > 0:
            status_text = "疑似违规 %d/%d" % (max_confirm, self.confirm_frames)
        else:
            status_text = "合规"

        info = {
            "persons": persons,
            "counts": counts,
            "violations": violations,
            "unsafe": counts[STATUS_UNSAFE],
            "partial": counts[STATUS_PARTIAL],
            "safe": counts[STATUS_SAFE],
            "worst": worst,
            "alarm": any_alarm,
            "new_alert": new_alert,
            "alarm_persons": alarm_persons,
            "tracks": len(self.tracks),
            "status_text": status_text,
        }
        self.last_info = info
        return info

    def status_text(self):
        return self.last_info.get("status_text", "无人")