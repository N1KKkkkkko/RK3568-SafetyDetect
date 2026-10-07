#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 端离线送测（一阶段 own_best）：headcut ONNX + onnxruntime，跑和板端同一套解码/判定。

为什么有用：i8 的 `.rknn` 只能在板端跑（PC 模拟器加载不了预编译模型），但 headcut 的
**ONNX** 在 PC 上能跑，且输出结构和 RKNN 完全一样
（box_prior (1,64,8400) + class scores (1,4,8400)）。本脚本直接调用板端同一份
`headcut_decode.analyze_frame()`，所以 PC 上调好的阈值可以原样搬到板子配置里。

三种输入：
  --img    单张图片
  --dir    一个目录（jpg/jpeg/png 批量）
  --video  视频文件（逐帧处理，可选输出标注视频和逐帧 CSV）

一阶段判定（own_best，4 类 person/helmet/no_helmet/vest）：
  一次推理出人框 + 帽/衣框，按部位区域（头盔看头部、反光衣看躯干）归属装备：
  同时匹配到帽+衣=SAFE，只匹配到一个=PARTIAL，都没匹配到=UNSAFE。no_helmet 不参与判定。

依赖（PC 端，不是板端）：pip install onnxruntime opencv-python numpy
一般不需要直接调用本脚本，用仓库根目录的 run_pc.sh 即可。
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "app"))

from headcut_decode import (letterbox, analyze_frame, draw_gear,
                            draw_gear_boxes)
from safety_rules import SafetyMonitor


def infer(sess, bgr, size):
    """letterbox + BGR->RGB + 归一化 + 推理，返回 (outputs, ratio, pad)，和板端一致。"""
    pre, ratio, pad = letterbox(bgr, (size, size))
    x = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    x = x.astype(np.float32) / 255.0
    return sess.run(None, {"images": x}), ratio, pad


def open_writer(path, fps, size):
    """按可播性优先级挑编码器：avc1(H.264) -> MJPG -> mp4v。"""
    for cc in ("avc1", "MJPG", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w, cc
        w.release()
    return None, None


def detect(sess, frame, args):
    """一帧：推理 + 后处理（坐标还原、人框过滤、装备归属）。"""
    outs, ratio, pad = infer(sess, frame, args.imgsz)
    return analyze_frame(
        outs, ratio, pad, frame.shape[:2],
        conf_person=args.conf_person, conf_gear=args.conf_gear,
        iou=args.iou, img_size=args.imgsz, nc=args.nc,
        overlap_thr=args.gear_overlap, min_person_area=args.min_person_area,
        max_persons=args.max_persons)


def run_video(sess, args):
    """视频模式：逐帧处理，状态机跨帧累计（和板端一致），可输出标注视频 + 逐帧 CSV。"""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("ERROR: 打不开视频", args.video)
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 121.0):
        fps = 25.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if w0 <= 0 or h0 <= 0:
        print("ERROR: 读不到视频分辨率")
        return 2
    print("视频: %s  %dx%d  fps=%.2f  帧数=%d" % (args.video, w0, h0, fps, total))

    writer = None
    if args.out_video:
        writer, codec = open_writer(args.out_video, fps, (w0, h0))
        if writer is None:
            print("WARN: 无法创建输出视频", args.out_video)
            args.out_video = None
        else:
            print("输出视频编码:", codec, "->", args.out_video)

    fcsv = None
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        fcsv = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv.writer(fcsv).writerow(["frame", "person", "x1", "y1", "x2", "y2",
                                   "helmet_conf", "vest_conf", "status"])

    alert_on = tuple(s.strip() for s in args.alert_on.split(",") if s.strip())
    monitor = SafetyMonitor(confirm_frames=args.confirm_frames,
                            clear_frames=max(1, args.confirm_frames * 3),
                            alert_on=alert_on)
    totals = {"SAFE": 0, "PARTIAL": 0, "UNSAFE": 0}
    idx = 0
    t0 = time.time()
    ms = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        persons = detect(sess, frame, args)
        ms = (time.time() - t) * 1000
        info = monitor.update(persons, frame.shape[:2])
        for p in persons:
            draw_gear_boxes(frame, p)
            draw_gear(frame, p["box"], p["status"], p["helmet_conf"],
                      p["vest_conf"], p.get("track_id"))
            totals[p["status"]] = totals.get(p["status"], 0) + 1
            if fcsv:
                x1, y1, x2, y2 = [int(v) for v in p["box"]]
                csv.writer(fcsv).writerow([idx, p.get("track_id"), x1, y1, x2, y2,
                                           round(p["helmet_conf"], 4),
                                           round(p["vest_conf"], 4), p["status"]])
        # 左上角进度（ASCII，避免 cv2 画不了中文）
        cv2.putText(frame, "%s  #%d  %.0f ms" % (info.get("status_short", ""), idx, ms),
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if info["alarm"] else (0, 255, 0), 2)
        if writer is not None:
            writer.write(frame)
        idx += 1
        if idx % 50 == 0:
            print("  已处理 %d/%s 帧  %s" % (idx, total or "?", info["status_text"]))
        if args.max_frames and idx >= args.max_frames:
            print("  达到 --max-frames %d，提前结束" % args.max_frames)
            break

    cap.release()
    if writer is not None:
        writer.release()
    if fcsv:
        fcsv.close()
    dt = time.time() - t0
    print("处理完成: %d 帧, 用时 %.1fs (%.1f FPS)" % (idx, dt, idx / max(dt, 1e-6)))
    print("合计: SAFE=%d PARTIAL=%d UNSAFE=%d"
          % (totals["SAFE"], totals["PARTIAL"], totals["UNSAFE"]))
    if args.out_video:
        print("标注视频:", args.out_video)
    if args.out_csv:
        print("逐帧 CSV:", args.out_csv)
    return 0


def run_images(sess, args):
    """图片模式：--img 单张 / --dir 批量，逐张独立判定（不跨图累计）。"""
    files = [args.img] if args.img else sorted(
        sum((glob.glob(os.path.join(args.dir or "", e))
             for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if not files:
        print("ERROR: 没有输入（用 --img / --dir / --video）")
        return 2

    alert_on = tuple(s.strip() for s in args.alert_on.split(",") if s.strip())
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, "pc_check_result.csv") if out_dir else None
    fcsv = open(report, "w", newline="", encoding="utf-8") if report else None
    if fcsv:
        csv.writer(fcsv).writerow(["image", "person", "x1", "y1", "x2", "y2",
                                   "helmet_conf", "vest_conf", "status"])

    totals = {"SAFE": 0, "PARTIAL": 0, "UNSAFE": 0}
    for path in files:
        frame = cv2.imread(path)
        if frame is None:
            print("无法读取", path)
            continue
        persons = detect(sess, frame, args)
        monitor = SafetyMonitor(confirm_frames=args.confirm_frames, clear_frames=1,
                                alert_on=alert_on)
        info = monitor.update(persons, frame.shape[:2])
        for pi, p in enumerate(persons):
            draw_gear_boxes(frame, p)
            draw_gear(frame, p["box"], p["status"], p["helmet_conf"],
                      p["vest_conf"], p.get("track_id"))
            totals[p["status"]] = totals.get(p["status"], 0) + 1
            if fcsv:
                x1, y1, x2, y2 = [int(v) for v in p["box"]]
                csv.writer(fcsv).writerow([os.path.basename(path), pi, x1, y1, x2, y2,
                                           round(p["helmet_conf"], 4),
                                           round(p["vest_conf"], 4), p["status"]])
        print("%-40s 人数=%d 状态=%s %s" % (os.path.basename(path)[:40], len(persons),
                                            info["status_text"], info["counts"]))
        for p in persons:
            print("      #%-3s %-7s 帽=%.2f 衣=%.2f 装备框=%d"
                  % (p.get("track_id"), p["status"], p["helmet_conf"],
                     p["vest_conf"], p["n_boxes"]))
        if out_dir:
            cv2.imwrite(os.path.join(out_dir, "pc_" + os.path.basename(path)), frame)

    if fcsv:
        fcsv.close()
        print("报告:", report)
    print("合计: SAFE=%d PARTIAL=%d UNSAFE=%d"
          % (totals["SAFE"], totals["PARTIAL"], totals["UNSAFE"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description="SafeDetect PC 端 ONNX 一阶段送测")
    ap.add_argument("--onnx", required=True, help="一阶段 headcut ONNX（own_best_headcut.onnx）")
    ap.add_argument("--img", default=None, help="单张图片")
    ap.add_argument("--dir", default=None, help="目录（处理其中所有 jpg/jpeg/png）")
    ap.add_argument("--video", default=None, help="视频文件（逐帧处理）")
    ap.add_argument("--out-dir", default=None, help="图片模式输出目录（默认不落盘）")
    ap.add_argument("--out-video", default=None, help="视频模式：标注结果另存为视频")
    ap.add_argument("--out-csv", default=None, help="视频模式：逐帧每人状态写 CSV")
    ap.add_argument("--max-frames", type=int, default=0, help="视频模式只处理前 N 帧（0=全部）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--conf-person", type=float, default=0.35)
    ap.add_argument("--conf-gear", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--gear-overlap", type=float, default=0.5,
                    help="帽/衣框落在人框对应部位区域内的比例下限")
    ap.add_argument("--min-person-area", type=float, default=0.0025)
    ap.add_argument("--max-persons", type=int, default=0, help="0=不限")
    ap.add_argument("--confirm-frames", type=int, default=1,
                    help="连续几帧违规才告警；图片用 1，视频建议 5")
    ap.add_argument("--alert-on", default="UNSAFE,PARTIAL")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: 需要 onnxruntime（pip install onnxruntime）")
        return 2

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print("模型输出:", [(o.name, o.shape) for o in sess.get_outputs()])

    if args.video:
        return run_video(sess, args)
    return run_images(sess, args)


if __name__ == "__main__":
    sys.exit(main())
