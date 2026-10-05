#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 端离线自检（一阶段 own_best）：headcut ONNX + onnxruntime，跑和板端同一套解码/判定。

为什么有用：i8 的 `.rknn` 只能在板端跑（PC 模拟器加载不了预编译模型），但 headcut 的
**ONNX** 在 PC 上能跑，且输出结构和 RKNN 完全一样
（box_prior (1,64,8400) + class scores (1,4,8400)）。本脚本直接调用板端同一份
`headcut_decode.analyze_frame()`，所以 PC 上调好的阈值可以原样搬到板子配置里。

一阶段判定（own_best，4 类 person/helmet/no_helmet/vest）：
  一次推理出人框 + 帽/衣框，按"重合率"关联：同时匹配到帽+衣=SAFE，只匹配到一个=PARTIAL，
  都没匹配到=UNSAFE。no_helmet 不参与判定。

依赖（PC 端，不是板端）：pip install onnxruntime opencv-python numpy

用法：
  python3 tools/pc_onnx_check.py --onnx models/own_best_headcut.onnx \
      --dir ../Safe_Model_Trainning/construction-ppe/images/test --out-dir /tmp/safe_pc_out
  python3 tools/pc_onnx_check.py --onnx models/own_best_headcut.onnx --img test.jpg
"""
import argparse
import csv
import glob
import os
import sys

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


def main():
    ap = argparse.ArgumentParser(description="SafeDetect PC 端 ONNX 一阶段自检")
    ap.add_argument("--onnx", required=True, help="一阶段 headcut ONNX（own_best_headcut.onnx）")
    ap.add_argument("--img", default=None, help="单张图片")
    ap.add_argument("--dir", default=None, help="目录（处理其中所有 jpg/jpeg/png）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认不落盘）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nc", type=int, default=4)
    ap.add_argument("--conf-person", type=float, default=0.35)
    ap.add_argument("--conf-gear", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--gear-overlap", type=float, default=0.5,
                    help="帽/衣框落在人框内的重合率下限")
    ap.add_argument("--min-person-area", type=float, default=0.0025)
    ap.add_argument("--max-persons", type=int, default=0, help="0=不限")
    ap.add_argument("--confirm-frames", type=int, default=1,
                    help="单张图自检用 1（逐图独立）；跑视频帧序列时可调大")
    ap.add_argument("--alert-on", default="UNSAFE,PARTIAL")
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: 需要 onnxruntime（pip install onnxruntime）")
        return 2

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print("模型输出:", [(o.name, o.shape) for o in sess.get_outputs()])

    files = [args.img] if args.img else sorted(
        sum((glob.glob(os.path.join(args.dir or "", e))
             for e in ("*.jpg", "*.jpeg", "*.png")), []))
    if not files:
        print("ERROR: 没有输入图片（用 --img 或 --dir）")
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
        outs, ratio, pad = infer(sess, frame, args.imgsz)
        persons = analyze_frame(
            outs, ratio, pad, frame.shape[:2],
            conf_person=args.conf_person, conf_gear=args.conf_gear,
            iou=args.iou, img_size=args.imgsz, nc=args.nc,
            overlap_thr=args.gear_overlap, min_person_area=args.min_person_area,
            max_persons=args.max_persons)
        # 每张图是独立场景：各起一个状态机，人编号从 1 开始，不跨图累计（与板端 --dir 一致）
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


if __name__ == "__main__":
    sys.exit(main())
