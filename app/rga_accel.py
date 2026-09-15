# -*- coding: utf-8 -*-
"""RGA hardware letterbox/resize for RK3568 (librga im2d API via ctypes).
注：现在没有使用，因为RGA 2D硬件在板上实测比OpenCV慢，且RGB888虚拟地址可能走软件回退，导致插值差异。
把 letterbox 的缩放 + BGR->RGB 交给 Rockchip RGA 2D 硬件，
主循环省掉 cv2.resize / cvtColor 的 CPU 开销，同时减少 NPU 空等。

模式（fall_config.json 顶层 preprocess 或 --preprocess）：
  auto  默认：启动时实测 RGA 与 OpenCV 各跑几次，谁快用谁（板子 librga
        版本太老导致 RGA 反而慢时自动选 OpenCV，并打印原因）；
  rga   强制 RGA（不可用时降级并打印告警）；
  cv2   纯 OpenCV（等价旧行为）。

板上自检：
  python3 rga_accel.py --selftest

兼容性备注：板端实测为 librga API 1.3.0（2021 年老版本）。老/新版 rga_buffer_t
前 40 字节（vir_addr..format）布局一致，resize 只用到这些字段，因此本模块的
ctypes 结构对老库同样有效；老库在该栈上对 RGB888 虚拟地址可能走软件回退，
表现为慢 + 插值差异，由 auto 实测自动规避。
"""
__version__ = "1.1.0"

import ctypes
import ctypes.util
import os
import sys
import time

import cv2
import numpy as np

# rga.h 像素格式枚举（老/新 librga 均为 <<8 形式）
RK_FORMAT_BGR_888 = 0x7 << 8   # 0x700
RK_FORMAT_RGB_888 = 0x2 << 8   # 0x200

IM_INTERP_LINEAR = 1            # im2d_type.h IM_INTER_MODE::IM_INTERP_LINEAR
IM_STATUS_SUCCESS = 1           # im2d 返回正数=成功，负数=失败
# 个别精简 Python 没有 ctypes.RTLD_NOW（板端实测），用 None 表示不传 mode
_RTLD_NOW = getattr(ctypes, "RTLD_NOW", None)

_LIB_NAMES = ("librga.so", "librga.so.2", "librga.so.1", "librga.so.0")
_LIB_DIRS = (
    "/usr/lib",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib/arm-linux-gnueabihf",
    "/usr/local/lib",
    "/oem/usr/lib",
    "/vendor/usr/lib",
)


class RgaRect(ctypes.Structure):
    """im2d_type.h 的 im_rect"""
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int)]


class RgaColorkeyRange(ctypes.Structure):
    _fields_ = [("max", ctypes.c_int), ("min", ctypes.c_int)]


class RgaNn(ctypes.Structure):
    _fields_ = [("scale_r", ctypes.c_int), ("scale_g", ctypes.c_int),
                ("scale_b", ctypes.c_int), ("offset_r", ctypes.c_int),
                ("offset_g", ctypes.c_int), ("offset_b", ctypes.c_int)]


class RgaBuffer(ctypes.Structure):
    """im2d_type.h 的 rga_buffer_t（现代布局，96 字节）。

    resize 只用前 40 字节（vir_addr..format），老库（1.3.x，88 字节）这部分
    布局一致；尾部字段全 0，老库不会读超出自身结构的字节。
    """
    _fields_ = [
        ("vir_addr", ctypes.c_void_p),
        ("phy_addr", ctypes.c_void_p),
        ("fd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("wstride", ctypes.c_int),
        ("hstride", ctypes.c_int),
        ("format", ctypes.c_int),
        ("color_space_mode", ctypes.c_int),
        ("global_alpha", ctypes.c_int),   # union { int global_alpha; ... }
        ("rd_mode", ctypes.c_int),
        ("color", ctypes.c_int),
        ("colorkey_range", RgaColorkeyRange),
        ("nn", RgaNn),
        ("rop_code", ctypes.c_int),
        ("handle", ctypes.c_uint32),
    ]


def _find_lib():
    """按常见路径查找 librga.so，找不到返回 None。"""
    for name in _LIB_NAMES:
        short = name[3:] if name.startswith("lib") else name
        short = short.split(".so")[0]
        try:
            p = ctypes.util.find_library(short)
        except Exception:
            p = None
        if p:
            return p
    for d in _LIB_DIRS:
        for name in _LIB_NAMES:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def letterbox_cv2(frame, new_shape=(640, 640), color=(114, 114, 114)):
    """OpenCV 版 letterbox（与 headcut_decode.letterbox 完全一致），返回 (rgb, ratio, pad)。"""
    from headcut_decode import letterbox as _lb
    img, ratio, pad = _lb(frame, new_shape, color)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), ratio, pad


class RgaAccel:
    """librga im2d 封装：letterbox 缩放硬件化；auto 模式实测选快者。"""

    def __init__(self, mode="auto"):
        self.mode = mode
        self.lib = None
        self.reason = "未初始化"
        self._wrap = None
        self._resize = None
        self._cache = {}
        self.rga_frames = 0
        self.cv_frames = 0
        self.fail_count = 0
        self.auto_disabled = False
        self.auto_use_cv2 = False
        self.bench_rga_ms = None
        self.bench_cv2_ms = None
        self._load()
        if self.mode == "auto" and self.available:
            self._bench()

    # ---------------- 加载 ----------------
    def _load(self):
        path = _find_lib()
        if not path:
            self.reason = "未找到 librga.so"
            return
        try:
            if _RTLD_NOW is not None:
                lib = ctypes.CDLL(path, mode=_RTLD_NOW)
            else:
                lib = ctypes.CDLL(path)
        except OSError as e:
            self.reason = "加载 %s 失败: %s" % (path, e)
            return
        wrap = getattr(lib, "wrapbuffer_virtualaddr_t", None) or \
            getattr(lib, "wrapbuffer_virtualaddr", None)
        resize = getattr(lib, "imresize_t", None) or \
            getattr(lib, "imresize", None)
        if wrap is None or resize is None:
            self.reason = "%s 缺少 im2d 符号(wrapbuffer_virtualaddr/imresize)，版本过旧" % path
            return
        wrap.restype = RgaBuffer
        wrap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                         ctypes.c_int, ctypes.c_int, ctypes.c_int]
        resize.restype = ctypes.c_int
        resize.argtypes = [RgaBuffer, RgaBuffer, ctypes.c_double,
                           ctypes.c_double, ctypes.c_int, ctypes.c_int]
        self.lib = lib
        self._wrap = wrap
        self._resize = resize
        ver = "librga"
        q = getattr(lib, "querystring", None)
        if q is not None:
            try:
                q.restype = ctypes.c_char_p
                q.argtypes = [ctypes.c_int]
                v = q(1)  # RGA_VERSION
                if v:
                    ver = v.decode(errors="replace")
            except Exception:
                pass
        self.reason = "已加载 %s (%s)" % (path, ver)

    @property
    def available(self):
        return self.lib is not None

    @property
    def enabled(self):
        if self.auto_use_cv2:
            return False
        return (not self.auto_disabled) and self.mode != "cv2" and self.available

    def _fail(self, msg):
        self.fail_count += 1
        if self.fail_count == 10:
            self.auto_disabled = True
            print("预处理: RGA 连续失败(%s)，自动禁用并退回 OpenCV" % msg)
        return None

    def info(self):
        extra = ""
        if self.bench_rga_ms is not None:
            extra = " | 实测 RGA=%.1fms cv2=%.1fms -> %s" % (
                self.bench_rga_ms, self.bench_cv2_ms,
                "cv2" if self.auto_use_cv2 else "RGA")
        return "%s | 模式=%s | 硬件缩放可用=%s%s%s" % (
            self.reason, self.mode,
            "是" if self.enabled else "否(自动退回cv2)",
            " | 已自动禁用" if self.auto_disabled else "",
            extra)

    # ---------------- 核心 ----------------
    def _wrap_buf(self, arr, fmt):
        h, w = arr.shape[:2]
        return self._wrap(arr.ctypes.data_as(ctypes.c_void_p),
                          w, h, w, h, fmt)

    def _letterbox_rga(self, frame, new_shape=(640, 640), color=(114, 114, 114)):
        """RGA letterbox：返回 (rgb_uint8_contiguous, ratio, pad) 或 None。

        ratio/pad 与 headcut_decode.letterbox 完全一致，后续 scale_coords 无需改动。
        本方法不做 enabled 判断，供 auto 实测与公开 letterbox 复用。
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        try:
            h, w = frame.shape[:2]
            nh0, nw0 = int(new_shape[0]), int(new_shape[1])
            r = min(nh0 / h, nw0 / w)
            nw, nh = int(round(w * r)), int(round(h * r))
            dw, dh = (nw0 - nw) / 2.0, (nh0 - nh) / 2.0
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            if nw <= 0 or nh <= 0 or nw > nw0 or nh > nh0:
                return None
            # RGA RGB888 需要 4 字节行对齐（w*3 % 4 == 0）；不满足就走 cv2
            if (w % 4) != 0 or (nw % 4) != 0:
                return None
            frame = np.ascontiguousarray(frame)
            if self.auto_disabled:
                return None
            dst = self._get_dst(nh0, nw0, color)
            tmp = self._get_tmp(nh, nw)
            src_buf = self._wrap_buf(frame, RK_FORMAT_BGR_888)
            tmp_buf = self._wrap_buf(tmp, RK_FORMAT_BGR_888)
            ret = self._resize(src_buf, tmp_buf, 0.0, 0.0,
                               IM_INTERP_LINEAR, 1)
            if ret <= 0:
                return self._fail("imresize ret=%d" % ret)
            # 通道交换 BGR->RGB + 贴到灰色画布（一次内存拷贝，极快）
            dst[top:top + nh, left:left + nw] = tmp[:, :, ::-1]
            self.rga_frames += 1
            self.fail_count = 0
            return dst, r, (dw, dh)
        except Exception as e:
            return self._fail("%s" % e)

    def letterbox(self, frame, new_shape=(640, 640), color=(114, 114, 114)):
        """对外接口：enabled 时走 RGA，否则返回 None（上层自动退回 cv2）。"""
        if not self.enabled:
            return None
        return self._letterbox_rga(frame, new_shape, color)

    # ---------------- auto 实测 ----------------
    def _bench(self):
        """auto 模式：小图实测 RGA vs cv2，谁快用谁（避免老 librga 反效果）。"""
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        n = 5
        for _ in range(2):
            self._letterbox_rga(frame, (416, 416))
            letterbox_cv2(frame, (416, 416))
        t0 = time.time()
        for _ in range(n):
            self._letterbox_rga(frame, (416, 416))
        t_rga = (time.time() - t0) / n * 1000
        t0 = time.time()
        for _ in range(n):
            letterbox_cv2(frame, (416, 416))
        t_cv = (time.time() - t0) / n * 1000
        self.bench_rga_ms, self.bench_cv2_ms = t_rga, t_cv
        self.auto_use_cv2 = t_rga > t_cv * 0.95
        if self.auto_use_cv2:
            print("预处理: auto 实测 RGA %.1fms 慢于 cv2 %.1fms，选择 OpenCV" %
                  (t_rga, t_cv))
        else:
            print("预处理: auto 实测 RGA %.1fms 快于 cv2 %.1fms，选择 RGA" %
                  (t_rga, t_cv))

    # ---------------- 缓冲复用 ----------------
    def _get_dst(self, h, w, color):
        key = ("dst", h, w, color[0], color[1], color[2])
        buf = self._cache.get(key)
        if buf is None:
            buf = np.empty((h, w, 3), dtype=np.uint8)
            self._cache[key] = buf
        buf[:] = color
        return buf

    def _get_tmp(self, h, w):
        key = ("tmp", h, w)
        buf = self._cache.get(key)
        if buf is None:
            buf = np.empty((h, w, 3), dtype=np.uint8)
            self._cache[key] = buf
        return buf


# ---------------- 自检 ----------------
def _gradient_frame():
    xs = np.linspace(0, 255, 640, dtype=np.uint8)
    ys = np.linspace(0, 255, 480, dtype=np.uint8)
    f = np.empty((480, 640, 3), dtype=np.uint8)
    f[..., 0] = xs[None, :]   # B 横向渐变
    f[..., 1] = ys[:, None]   # G 纵向渐变
    f[..., 2] = 128           # R 常量
    return f


def _diff_stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return float(d.max()), float(d.mean())


def selftest(img_size=416):
    """板上自检：对比 RGA 与 OpenCV letterbox 输出（最大/平均像素差）并测耗时。"""
    print("[RGA] rga_accel v%s" % __version__)
    accel = RgaAccel("rga")
    print("[RGA] %s" % accel.info())
    if not accel.available:
        return 1
    for name, frame in (("梯度图", _gradient_frame()),
                        ("随机噪声", np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))):
        out = accel._letterbox_rga(frame, (img_size, img_size))
        if out is None:
            print("[RGA] %s: letterbox 调用失败，自动退回 cv2" % name)
            continue
        img_rgb, ratio, pad = out
        ref, r2, p2 = letterbox_cv2(frame, (img_size, img_size))
        mx, mn = _diff_stats(img_rgb, ref)
        print("[RGA] %s: 尺寸=%s ratio=%.4f pad=%s 最大像素差=%g 平均像素差=%.2f" %
              (name, img_rgb.shape, ratio, pad, mx, mn))
    # 耗时：RGA 全流程 vs cv2 全流程
    frame = _gradient_frame()
    for _ in range(5):
        accel._letterbox_rga(frame, (img_size, img_size))
        letterbox_cv2(frame, (img_size, img_size))
    n = 30
    t0 = time.time()
    for _ in range(n):
        accel._letterbox_rga(frame, (img_size, img_size))
    t_rga = (time.time() - t0) / n * 1000
    t0 = time.time()
    for _ in range(n):
        letterbox_cv2(frame, (img_size, img_size))
    t_cv = (time.time() - t0) / n * 1000
    print("[RGA] RGA letterbox %.2f ms/帧, cv2 对比 %.2f ms/帧" % (t_rga, t_cv))
    print("[RGA] 结论: %s" % ("建议用 cv2（RGA 更慢）" if t_rga > t_cv else "RGA 可用（更快或相当）"))
    return 0


if __name__ == "__main__":
    sys.exit(selftest())