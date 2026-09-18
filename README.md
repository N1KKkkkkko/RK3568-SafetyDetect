# 工地安全着装检测边缘AI终端 (RK3568)

基于 RK3568 的边缘 AI 安全监测系统，支持安全帽/反光安全衣检测与烟雾报警，全程本地推理，不上云保证隐私安全。

## ✨ 功能特性

- 🪖 安全帽 / 反光安全衣检测（YOLOv8n + NPU）
- 🎯 两阶段流水线：先找人，再对每个人单独判断装备
- 🧍 逐人判定 SAFE / PARTIAL / UNSAFE，画面按状态着色
- 📈 逐人跟踪 + 连续帧确认，抑制单帧误检
- 🔥 烟雾报警（MQ-2 传感器，与着装告警同一条通道）
- 📱 手机告警推送（截图 + 前后录像 + 一键看实时画面）
- 🖥️ 网页实时画面，手机浏览器直接打开（MJPEG）
- 🌐 远程访问（可选）：Tailscale 内网穿透，不在同一局域网也能收告警、看画面，不开放公网端口
- 🛡️ 无摄像头/掉线不退出，显示占位画面并推通知，插回自动恢复

## 🏗️ 技术架构

摄像头(USB UVC) →预处理(CPU) → RK3568 NPU推理 → 后处理(CPU) → 告警链路 → 手机

- 模型：YOLOv8n (人形检测) + YOLOv8n (安全装备，2 类)
- 部署：RKNN-Toolkit2 1.3.0, librknnrt 1.3.0（模型版本必须与板端运行时同代）
- 推理：NPU 负责卷积，CPU 负责 DFL 解码 / NMS / 坐标还原
- 告警：MQTT → Node-RED → ntfy；实时画面走板端 MJPEG
- 远程访问：Tailscale 虚拟 IP 由板端自动探测，随告警消息下发给 Node-RED；手机与板子登录同一账号即可直连（内网、外网通用）

## 📦 快速开始

### 环境要求

- **硬件**：Firefly ROC-RK3568-PC（设备树标识 `RK3568-ROC-PC HDMI`；4×Cortex-A55 / 4GB RAM / eMMC），USB UVC 摄像头
- **系统**：Ubuntu 20.04.6 LTS / Python 3.8.10 / 内核 4.19.232
- **NPU**：驱动 0.8.2 + librknnrt 1.3.0，配套 rknn-toolkit-lite2 2.3.2（镜像预装）
- **已预装 Python 库**：numpy 1.24.4、OpenCV 5.0.0.93、paho-mqtt 2.1.0

> 板端根文件系统是 overlayroot（底层 `/root-ro` 只读 2.5G + 可写层在 `/userdata` 26G），
> `df -h /` 实测剩余约 18G，项目与告警产物写在用户目录即可；截图/录像按 `alerts.max_records`
> （默认 100 对）自动清理最旧记录。内核较老（4.19）时 RGA 硬件缩放可能慢于 OpenCV，
> 程序启动会实测后自动选择，无需手动干预。

### 安装

```bash
# 板端镜像通常已装好 Python 依赖，只需确认下面这几个系统命令
sudo apt install -y gpiod v4l-utils mosquitto mosquitto-clients
sudo apt install -y ffmpeg            # 可选：告警录像转 H.264

# 若换到缺依赖的机器，再补 Python 依赖
pip3 install -r requirements.txt
sudo apt install -y gpiod mosquitto mosquitto-clients ffmpeg
```

### 配置

```bash
cp config/safe_config_git.json config/safe_config.json   # 首次部署：生成私有配置
```

真实部署值都写在 `config/safe_config.json`：本机对外 IP、摄像头节点、传感器 GPIO/ADC 接线与报警阈值、
MQTT 账号、端口等。它在 `.gitignore` 里，不会上传；`config/safe_config_git.json` 是可上传的公开模板，
两者字段一致。程序优先读私有配置，没有（例如刚 clone）就自动退回模板，所以不配也能先跑起来。

### 运行

```bash
./run.sh --source /dev/video0 --noshow        # USB 摄像头（节点按实际填）
./run.sh --img test.jpg --out out.jpg         # 单张图片调试（打印每人置信度）
./run.sh --dir ~/frames --out-dir ~/out       # 批量图片 + CSV 报告
./run.sh --source ~/site.mp4                  # 回放视频文件（按原帧率）
./run.sh --bench 30                           # 分阶段性能测试
```

实时画面：`http://<板子IP>:8090/`　手机告警：下载ntfy 订阅主题 `safe_cam1` 服务链接地址 `http://<板子IP>`

**远程访问（可选）**：同一局域网内直接访问即可，不需要额外组件。要出门也能看时，
板子和手机都装 [Tailscale](https://tailscale.com/)（登录同一账号）：

```bash
curl -fsSL https://tailscale.com/install.sh | sh   # 板端安装
sudo tailscale up                                  # 打印授权链接，浏览器登录同账号
tailscale ip -4                                    # 板子的虚拟 IP，手机用它替换 <板子IP>
```

装好后，实时画面 / 告警截图 / 录像链接都走这个虚拟 IP，公网不用开任何端口。
没装 Tailscale 程序也照常运行 —— 程序会自动探测可用地址（优先 Tailscale，其次局域网 IP）。

告警推送需要 mosquitto / Node-RED / ntfy，`deploy/install_notify_stack.sh` 可一次装好，
systemd 单元与 Node-RED 流程文件都在 `deploy/` 下。

## 📊 性能指标

| 指标 | 数值 (RK3568, int8, 640 输入) |
| --- | --- |
| 人形检测 | ~80 ms |
| 装备检测 | ~75 ms / 人 |
| 后处理解码 | 3 ~ 5 ms |
| 画面无人 | ~82 ms（≈12 FPS） |
| 画面 1 人 | ~158 ms（≈6 FPS） |

## 📁 目录结构

```
├── app/          # 源码：入口、两级解码、合规状态机、推流、告警、录像、传感器
├── config/       # 运行配置：safe_config_git.json(公开模板,可上传) / safe_config.json(私有,不入库)
├── models/       # RKNN 模型与版本说明
├── deploy/       # systemd 单元、udev 规则、Node-RED 流程、部署脚本
└── alerts/       # 告警截图与录像（运行产物，不入库）
```

## 📝 项目背景

工地要求"戴安全帽 + 穿反光安全衣"，人工巡查效率低、覆盖有限。本项目把检测、判定、告警、
远程查看做成一个能在边缘板上长期无人值守运行的终端，视频与告警数据都留在本地，
满足工业场景的隐私要求。

## 🔗 相关链接

- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) —— 模型转换工具包
- [Firefly Wiki](https://wiki.t-firefly.com/zh_CN/ROC-RK3568-PC/started.html) —— 开发板相关文档
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) —— YOLO 系列模型开发
- [Tailscale](https://tailscale.com/) —— 内网穿透
- [ntfy](https://ntfy.sh/) —— 手机推送

## 🧠 模型说明

本项目采用两级检测架构，仅供学习与交流，非商业用途，使用的第三方模型遵循其原始许可证。

- **一级模型**：基于 [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics)（COCO 预训练权重）。
- **二级模型**：基模来自开源项目 [Safety-Vest-and-Helmet-Detection](https://github.com/ADiTyaRaj8969/Safety-Vest-and-Helmet-Detection)。

---

## 实物图
<img width="1280" height="1068" alt="789e71524f091cfd9da9321d0ee1e9c4_720" src="https://github.com/user-attachments/assets/9af024b3-c68e-48ed-8cd2-d6eded7a3c2c" />

---
联系邮箱：1995466@qq.com
