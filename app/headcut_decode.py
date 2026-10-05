# -*- coding: utf-8 -*-
"""headcut YOLOv8 RKNN 解码 + PPE 合规判定。

所有模型的输出结构一样（headcut 砍掉了 rknn-toolkit2 1.3.0 编不了的 DFL/dist2bbox
尾巴，板端 Python 补回）：
  outputs[0] box DFL logits  (1, 64, anchors)   4 边 x 16 bins
  outputs[1] class scores    (1, nc, anchors)   已过 sigmoid

本包用【一阶段】模型 own_best（nc=4）：
  0 person / 1 helmet / 2 no_helmet / 3 vest
一次推理同时出人框和帽/衣框，再用"重合率"把人框和帽/衣框关联起来判定：
  同时匹配到 helmet 和 vest -> SAFE
  只匹配到其中一个        -> PARTIAL
  都没匹配到              -> UNSAFE
**no_helmet 不参与判定**（按需求忽略；没戴帽的人靠"匹配不到 helmet"判出来）。
"""
import cv2
import numpy as np

STATUS_SAFE = "SAFE"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNSAFE = "UNSAFE"

# 状态 -> BGR 颜色（绿 / 黄 / 红）
STATUS_COLOR = {
    STATUS_SAFE: (0, 255, 0),
    STATUS_PARTIAL: (0, 255, 255),
    STATUS_UNSAFE: (0, 0, 255),
}

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


# ---------------- 三态判定 / 画图 ----------------
def status_color(status):
    return STATUS_COLOR.get(status, (200, 200, 200))


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


# ======================================================================
# 一阶段模型 own_best（nc=4）专用：一次推理出人 + 帽 + 衣，重合率关联
# ======================================================================
OWN_NC = 4
OWN_CLASS_PERSON = 0
OWN_CLASS_HELMET = 1
OWN_CLASS_NO_HELMET = 2   # 按需求忽略，不参与判定
OWN_CLASS_VEST = 3

# 参与判定的类别（no_helmet 明确排除）
OWN_USE_CLASSES = (OWN_CLASS_PERSON, OWN_CLASS_HELMET, OWN_CLASS_VEST)


def decode_det(outputs, conf=0.25, iou=0.45, img_size=640, nc=OWN_NC,
               classes=None):
    """在给定类别子集内做 argmax 解码（返回 boxes xyxy 模型尺度, scores, classes）。

    先截取 classes 指定的列再做 argmax：这样排除 no_helmet 之后，
    一个既有 helmet 分、又有 no_helmet 分的锚点仍会按 helmet 参与判定，
    而不是被 no_helmet 抢占后丢掉。
    """
    scores = np.asarray(outputs[1], dtype=np.float32).reshape(nc, -1).T
    if classes is not None:
        sub_idx = np.asarray(classes, dtype=np.int64)
        sub = scores[:, sub_idx]
        cls = sub_idx[sub.argmax(axis=1)]
        sc = sub.max(axis=1)
    else:
        cls = scores.argmax(axis=1)
        sc = scores.max(axis=1)

    keep = sc >= conf
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

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


def containment_ratio(inner, outer):
    """重合率 = inner 框落在 outer 框内的面积 / inner 框面积。

    人框和帽子/衣服框大小悬殊，用 IoU 会趋近于 0（大框包小框时 IoU≈小/大），
    所以这里用"小框被大框盖住的比例"来判"这件装备是不是属于这个人"。
    """
    ix1 = max(float(inner[0]), float(outer[0]))
    iy1 = max(float(inner[1]), float(outer[1]))
    ix2 = min(float(inner[2]), float(outer[2]))
    iy2 = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, float(inner[2]) - float(inner[0])) * \
        max(0.0, float(inner[3]) - float(inner[1]))
    return inter / (area + 1e-9)


def match_gear_to_persons(pboxes, pscores, gboxes, gscores, gclasses,
                          overlap_thr=0.5):
    """把每个帽/衣框分配给"重合率最高、且达到阈值"的那个人，输出三态结论。

    返回的每个人是一个 dict：
      box / score / status / helmet_conf / vest_conf / n_boxes
    另外带 helmet_box / vest_box（命中的装备框，便于画图/排查），没有则为 None。
    """
    persons = []
    for b, s in zip(pboxes, pscores):
        persons.append({"box": [float(v) for v in b], "score": float(s),
                        "helmet_conf": 0.0, "vest_conf": 0.0,
                        "helmet_box": None, "vest_box": None, "n_boxes": 0})

    for gb, gs, gc in zip(gboxes, gscores, gclasses):
        best_i, best_r = -1, 0.0
        for i, p in enumerate(persons):
            r = containment_ratio(gb, p["box"])
            if r > best_r:
                best_r, best_i = r, i
        if best_i < 0 or best_r < overlap_thr:
            continue                      # 不属于任何人（或与所有框重合太低）的装备框丢弃
        p = persons[best_i]
        p["n_boxes"] += 1
        if int(gc) == OWN_CLASS_HELMET and float(gs) > p["helmet_conf"]:
            p["helmet_conf"] = float(gs)
            p["helmet_box"] = [float(v) for v in gb]
        elif int(gc) == OWN_CLASS_VEST and float(gs) > p["vest_conf"]:
            p["vest_conf"] = float(gs)
            p["vest_box"] = [float(v) for v in gb]

    for p in persons:
        if p["helmet_conf"] > 0.0 and p["vest_conf"] > 0.0:
            p["status"] = STATUS_SAFE
        elif p["helmet_conf"] > 0.0 or p["vest_conf"] > 0.0:
            p["status"] = STATUS_PARTIAL
        else:
            p["status"] = STATUS_UNSAFE
    return persons


def analyze_frame(outputs, ratio, pad, frame_shape,
                  conf_person=0.35, conf_gear=0.25, iou=0.45, img_size=640,
                  nc=OWN_NC, overlap_thr=0.5, min_person_area=0.0025,
                  max_persons=0, min_side=16):
    """一阶段完整后处理：解码 -> 坐标还原 -> 人框过滤 -> 重合率关联 -> 三态。

    这是板端主程序和 PC 端 tools/pc_onnx_check.py 共用的同一份逻辑，
    保证"PC 上调好的阈值能原样搬到板子"。
    返回 persons 列表（box 已还原到原图坐标）。
    """
    boxes, scores, cls = decode_det(outputs, conf=min(conf_person, conf_gear),
                                    iou=iou, img_size=img_size, nc=nc,
                                    classes=list(OWN_USE_CLASSES))
    cls = cls.astype(np.int32)
    m_person = cls == OWN_CLASS_PERSON
    m_gear = (cls == OWN_CLASS_HELMET) | (cls == OWN_CLASS_VEST)
    psel = m_person & (scores >= conf_person)
    gsel = m_gear & (scores >= conf_gear)

    pboxes = scale_coords(boxes[psel], ratio, pad, frame_shape)
    pscores = scores[psel]
    gboxes = scale_coords(boxes[gsel], ratio, pad, frame_shape)
    gscores = scores[gsel]
    gclasses = cls[gsel]

    pboxes, pscores = filter_person_boxes(pboxes, pscores, frame_shape,
                                          min_area=min_person_area,
                                          max_persons=max_persons,
                                          min_side=min_side)
    return match_gear_to_persons(pboxes, pscores, gboxes, gscores, gclasses,
                                 overlap_thr)


def draw_gear_boxes(img, person, thickness=1):
    """画命中的装备框：安全帽青色、反光衣品红（只用于排查，不影响判定）。"""
    hb = person.get("helmet_box")
    vb = person.get("vest_box")
    if hb:
        cv2.rectangle(img, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])),
                      (255, 200, 0), thickness)
    if vb:
        cv2.rectangle(img, (int(vb[0]), int(vb[1])), (int(vb[2]), int(vb[3])),
                      (255, 0, 255), thickness)
