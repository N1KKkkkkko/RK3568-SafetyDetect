# -*- coding: utf-8 -*-
"""headcut YOLOv8 RKNN 解码（本包两级模型共用）。

两个模型的输出结构完全一样，只是类别数 nc 不同：
  一级  yolov8n_headcut.rknn         nc=80 (COCO)  只取 class 0 = person 找人体框
  二级  yolov8n_safe_headcut_i8.rknn nc=3          安全帽 / 安全衣判定

headcut 砍掉了 rknn-toolkit2 1.3.0 编不了的 DFL/dist2bbox 尾巴，板端 Python 补：
  outputs[0] box DFL logits  (1, 64, anchors)   4 边 x 16 bins
  outputs[1] class scores    (1, nc, anchors)   已过 sigmoid

二级类别（SafeDetect_Model_trans/labels.txt）：
  0 No Vest             未穿安全衣
  1 person_with_helmet  人+安全帽
  2 person_with_vest    人+安全衣

三态判定（与原项目 README / rknn_two_stage_test.py 一致）：
  同时命中 1 和 2 -> SAFE；只命中其一 -> PARTIAL；都没命中 -> UNSAFE。
"""
import cv2
import numpy as np

# ---- 二级模型类别 ----
GEAR_CLASS_NO_VEST = 0
GEAR_CLASS_HELMET = 1
GEAR_CLASS_VEST = 2
GEAR_NC = 3

STATUS_SAFE = "SAFE"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNSAFE = "UNSAFE"

# 状态 -> BGR 颜色（与原项目 README 配色一致：绿 / 黄 / 红）
STATUS_COLOR = {
    STATUS_SAFE: (0, 255, 0),
    STATUS_PARTIAL: (0, 255, 255),
    STATUS_UNSAFE: (0, 0, 255),
}

PERSON_NC = 80        # COCO
PERSON_CLASS_ID = 0   # COCO class 0 = person

_GRID_CACHE = {}
_DFL_W = np.arange(16, dtype=np.float32)


# ---------------- 预处理 / 坐标还原 ----------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """等比缩放 + 灰边填充；返回 (图像, ratio, (pad_x, pad_y))，与原项目一致。"""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def scale_coords(coords, ratio, pad, img0_shape):
    """letterbox 坐标 -> 原图坐标（xyxy，越界裁剪）。"""
    c = np.array(coords, dtype=np.float32)
    if len(c) == 0:
        return c
    c[:, [0, 2]] -= pad[0]
    c[:, [1, 3]] -= pad[1]
    c[:, :4] /= ratio
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, img0_shape[1])
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, img0_shape[0])
    return c


# ---------------- 解码 ----------------
def grid_arrays(img_size=640):
    """锚点网格 (ax, ay, st)，按 img_size 缓存（避免每帧重建 8400 个坐标）。"""
    g = _GRID_CACHE.get(img_size)
    if g is None:
        axs, ays, sts = [], [], []
        for st in (8, 16, 32):
            n = img_size // st
            sy, sx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            axs.append(sx.reshape(-1).astype(np.float32))
            ays.append(sy.reshape(-1).astype(np.float32))
            sts.append(np.full(n * n, st, np.float32))
        g = (np.concatenate(axs), np.concatenate(ays), np.concatenate(sts))
        _GRID_CACHE[img_size] = g
    return g


def nms_per_class(boxes, scores, classes, iou_thres=0.45, max_candidates=300):
    """逐类 NMS（同类别才互相抑制）。boxes 为 xyxy。

    性能说明（板端实测）：i8 量化的一级 COCO 模型每帧会产出几十上百个低分候选框，
    原来用 Python 双重循环做 NMS，这种量级要 14~20ms —— 占掉整帧预算的 20%
    （日志里"无人时解码 18~25ms"就是它）。
    现在按分数降序、逐框用 numpy 向量化抑制（语义与逐框循环一致，结果不变），
    并先用 max_candidates 截断到分数最高的前 N 个，避免极端情况 O(K²) 爆炸。
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))
    boxes = np.asarray(boxes, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int32)

    order = np.argsort(-scores, kind="stable")
    if max_candidates and order.size > max_candidates:
        order = order[:max_candidates]
    boxes, scores, classes = boxes[order], scores[order], classes[order]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep = np.ones(scores.shape[0], dtype=bool)
    for i in range(scores.shape[0]):
        if not keep[i]:
            continue
        rest = np.nonzero(keep[i + 1:])[0] + (i + 1)
        if rest.size == 0:
            continue
        same = classes[rest] == classes[i]
        if not same.any():
            continue
        r = rest[same]
        ix1 = np.maximum(x1[i], x1[r])
        iy1 = np.maximum(y1[i], y1[r])
        ix2 = np.minimum(x2[i], x2[r])
        iy2 = np.minimum(y2[i], y2[r])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[r] - inter + 1e-9)
        keep[r[iou > iou_thres]] = False

    sel = np.nonzero(keep)[0]
    return boxes[sel].copy(), scores[sel].copy(), classes[sel].copy()

def decode(outputs, conf=0.25, iou=0.45, img_size=640, nc=3, classes=None):
    """解码 headcut 输出，返回 (boxes[N,4] xyxy 像素, scores[N], classes[N])。

    classes: 只保留这些类别 id（如一级只取 [0] = person）；None 表示全部保留。
    优化：先用已有的 sigmoid 分数按 conf 过滤，只对高分锚点做 DFL 软max
    （全量 8400 锚点软max 在板端是毫秒级开销大头）。
    """
    scores = np.asarray(outputs[1], dtype=np.float32).reshape(nc, -1).T
    cls = scores.argmax(axis=1)
    sc = scores.max(axis=1)
    keep = sc >= conf
    if classes is not None:
        keep &= np.isin(cls, np.asarray(classes))
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)

    ax, ay, st = grid_arrays(img_size)
    prior = np.asarray(outputs[0], dtype=np.float32).reshape(4, 16, -1)[:, :, idx]
    e = np.exp(prior - prior.max(axis=1, keepdims=True))
    s = e / e.sum(axis=1, keepdims=True)
    l, t, r, b = np.tensordot(s, _DFL_W, axes=([1], [0]))
    cx = (ax[idx] + 0.5 + (r - l) / 2) * st[idx]
    cy = (ay[idx] + 0.5 + (b - t) / 2) * st[idx]
    bw = (l + r) * st[idx]
    bh = (t + b) * st[idx]
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return nms_per_class(xyxy, sc[idx], cls[idx], iou)


def decode_persons(outputs, conf=0.35, iou=0.45, img_size=640,
                   nc=PERSON_NC, person_class=PERSON_CLASS_ID):
    """一级模型：只取 person 类的框，返回 (boxes[N,4] xyxy, scores[N])。"""
    return decode(outputs, conf, iou, img_size, nc, classes=[person_class])


def decode_gear(outputs, conf=0.25, iou=0.45, img_size=640, nc=GEAR_NC):
    """二级模型：安全帽/安全衣框，返回 (boxes, scores, classes)。"""
    return decode(outputs, conf, iou, img_size, nc, classes=None)


def filter_person_boxes(boxes, scores, frame_shape=None, min_area=0.0025,
                        max_persons=0, min_side=16):
    """人体框过滤 + 排序：丢掉过小/贴边的框，按面积从大到小排序。

    min_area: 框面积占整幅画面比例下限（滤远处噪点，默认 0.25%）
    max_persons: >0 时只保留面积最大的前 N 个人（限制二级推理次数）
    """
    keep = []
    h, w = (frame_shape[0], frame_shape[1]) if frame_shape else (0, 0)
    for i, b in enumerate(boxes):
        bw = float(b[2] - b[0]); bh = float(b[3] - b[1])
        if bw < min_side or bh < min_side:
            continue
        if h and w and (bw * bh) < min_area * h * w:
            continue
        keep.append(i)
    if not keep:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    boxes = boxes[keep]
    scores = scores[keep]
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    if max_persons and max_persons > 0:
        order = order[:max_persons]
    return boxes[order], scores[order]


def crop_person(frame, box, pad_ratio=0.08):
    """按比例外扩裁剪人体，返回 (crop, (x1, y1))；裁剪失败返回 (None, None)。"""
    h0, w0 = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pw = max(4, int((x2 - x1) * pad_ratio))
    ph = max(4, int((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pw); cy1 = max(0, y1 - ph)
    cx2 = min(w0, x2 + pw); cy2 = min(h0, y2 + ph)
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None, None
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1)


# ---------------- 三态判定 / 画图 ----------------
def gear_status(boxes, scores, classes, min_conf=0.0):
    """把一个人体裁剪里的二级框聚合成三态结论。

    返回 dict: status / helmet_conf / vest_conf / no_vest_conf / n_boxes
    """
    helmet = 0.0
    vest = 0.0
    no_vest = 0.0
    n = 0
    for i in range(len(scores)):
        s = float(scores[i])
        if s < min_conf:
            continue
        n += 1
        c = int(classes[i])
        if c == GEAR_CLASS_HELMET:
            helmet = max(helmet, s)
        elif c == GEAR_CLASS_VEST:
            vest = max(vest, s)
        elif c == GEAR_CLASS_NO_VEST:
            no_vest = max(no_vest, s)

    if helmet > 0.0 and vest > 0.0:
        status = STATUS_SAFE
    elif helmet > 0.0 or vest > 0.0:
        status = STATUS_PARTIAL
    else:
        status = STATUS_UNSAFE
    return {"status": status, "helmet_conf": helmet, "vest_conf": vest,
            "no_vest_conf": no_vest, "n_boxes": n}


def status_color(status):
    return STATUS_COLOR.get(status, (200, 200, 200))


def draw_person(img, box, status, score=0.0, thickness=2):
    """一级人体框（灰）——用于确认"人找到了"，二级再按状态着色覆盖。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), thickness)
    cv2.putText(img, "person %.2f" % score, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def draw_gear(img, box, status, helmet_conf=0.0, vest_conf=0.0, track_id=None,
              thickness=2):
    """按三态着色画人体框 + 标签（SAFE 绿 / PARTIAL 黄 / UNSAFE 红）。"""
    x1, y1, x2, y2 = [int(v) for v in box]
    color = status_color(status)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    tag = "#%d " % track_id if track_id is not None else ""
    label = "%s%s H=%.2f V=%.2f" % (tag, status, helmet_conf, vest_conf)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1, th + 6)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, label, (x1 + 4, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)