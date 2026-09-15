# -*- coding: utf-8 -*-
"""
MJPEG 网页预览服务（multipart/x-mixed-replace）。端口由主程序 --stream 传入（本项目用 8090）。

JPEG 尺寸/质量可调（默认 480 宽、质量 60）：--stream-width / --stream-quality。
"""
import http.server
import socket
import socketserver
import threading
import time

import cv2


class StreamHandler(http.server.BaseHTTPRequestHandler):
    latest_jpeg = None     # 最新一帧的 JPEG 字节
    seq = 0                # 帧序号：变了才推送（关键优化）
    clients = 0            # 正在看 /stream 的连接数（无人观看时跳过编码）

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = ("<html><head><title>SafeDetect - 工地安全监控</title>"
                    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                    "</head><body style=\"margin:0;background:#111\">"
                    "<img src=\"/stream\" style=\"width:100%\">"
                    "</body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                # 关闭 Nagle：小块数据立即发出，不攒着等 ACK
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            StreamHandler.clients += 1
            last_sent = -1
            try:
                while True:
                    if StreamHandler.seq != last_sent and StreamHandler.latest_jpeg:
                        last_sent = StreamHandler.seq
                        jpg = StreamHandler.latest_jpeg
                        try:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                    else:
                        # 没有新帧：短睡等待（不写任何数据，避免把链路堵死）
                        time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                StreamHandler.clients = max(0, StreamHandler.clients - 1)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def get_lan_ip(probe=None):
    """获取本机局域网 IP，用于打印预览地址。

    probe 是"探测目标"（UDP connect 只挑路由、不发包），可以是单个地址或地址列表；
    来自 config network.lan_probe，换现场网络不用改代码。默认 1.1.1.1（仅用于选路）。
    """
    probes = list(probe) if isinstance(probe, (list, tuple)) else []
    if not probes:
        probes = [probe or "1.1.1.1"]
    for p in probes:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((str(p), 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        finally:
            s.close()
    return "127.0.0.1"


class ReusableStreamServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_stream(port, lan_probe=None):
    """启动 MJPEG 预览服务；端口被占时自动顺延，最多试 20 个。"""
    httpd = None
    p = port
    for p in range(port, port + 20):
        try:
            httpd = ReusableStreamServer(("0.0.0.0", p), StreamHandler)
            break
        except OSError:
            httpd = None
            continue
    if httpd is None:
        print("警告: 端口 %d~%d 都被占用，预览服务未启动" % (port, port + 19))
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("实时预览服务已启动: http://%s:%d/  浏览器打开即可看到标注画面" % (get_lan_ip(lan_probe), p))


def stream_publish_loop(stream_q, width=480, quality=60):
    """后台线程：把标注帧缩放+JPEG 编码，供浏览器 MJPEG 流使用，不卡主循环。

    - 没有浏览器在看 /stream 时跳过缩放+编码（省 CPU）；
    - 编码完成后 latest_jpeg + seq 一起更新，客户端据此只推新帧（见文件头说明）。
    """
    width = int(width or 0)
    quality = int(quality or 60)
    while True:
        frame = stream_q.get()
        if StreamHandler.clients <= 0:
            continue   # 无人观看：不编码
        if width and frame.shape[1] > width:
            h, w = frame.shape[:2]
            nh = int(round(h * float(width) / w))
            frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            StreamHandler.latest_jpeg = buf.tobytes()
            StreamHandler.seq += 1