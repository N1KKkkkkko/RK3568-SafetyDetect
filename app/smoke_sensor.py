#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQ-2 烟雾/燃气传感器脚本

接线与阈值全部走配置，不写死在代码里（本机实际接线填在私有 config/safe_config.json 的 smoke 节）：
    smoke.do.gpiochip / smoke.do.line / smoke.do.active_low     # DO 数字量 -> 某个 GPIO
    smoke.ao.enable / smoke.ao.iio_path                         # AO 模拟量 -> 某个 ADC 原始值节点
    smoke.ao.vref / smoke.ao.bits                               # ADC 参考电压与位数
    smoke.ao.divider                                            # 分压比（ADC 电压 -> 传感器电压）
    smoke.ao.alarm_raw / smoke.ao.alarm_hysteresis              # 按 ADC 原始值报警的阈值与迟滞

换板子/换引脚只改配置：DO 接的 GPIO 用 libgpiod 的 gpiochip+line 表示（RK 是 bank*32+index），
AO 接的 ADC 用 iio 原始值节点表示。都不填也能跑 —— 只凭 DO 数字量告警。

日志里的数值：DO 翻转会立刻打一行，读数每 smoke.log_interval 秒打一行（默认 10s，设 0 关闭）：
    journalctl -u safe-smoke -f

验证接线（换过板子/引脚后先跑一次，确认极性再开服务）：
    gpiodetect                              # 列出 gpiochip，确认配置里写的那个存在
    gpioinfo <gpiochip>                      # 看对应 line 的名字/占用情况
    python3 app/smoke_sensor.py --watch 20   # 实时打印 DO 电平与 AO 原始值；吹口烟看是否变化

MQTT  Topic ：
    safe/<camera_id>/alert            烟雾告警（event=smoke）
    safe/<camera_id>/smoke/value      DO + AO 遥测（retained）
    safe/<camera_id>/smoke/heartbeat  在线心跳（retained）
自检模式：
    持续20s，默认 0.5 秒一行
    sudo python3 ~/rk3568_camera/SafeDetect_v0.01/SafeDetect_V0.01/app/smoke_sensor.py --watch 20

"""
import argparse
import json
import os
import subprocess
import sys
import time

from config import pick_config            # 私有 safe_config.json 优先，缺失用公开模板

# ---------------- 默认配置（config 里没有的键才用这里） ----------------
DEFAULTS = {
    "camera_id": "cam1",
    "mqtt": {"broker": "", "port": 1883, "username": "", "password": "",
             "topic_prefix": "safe", "qos": 1, "heartbeat_interval": 30},
    "smoke": {
        "topic_prefix": "safe",
        # 数字量：GPIO1_A1；active_low=True 表示"低电平=检测到烟雾"
        "do": {"gpiochip": "gpiochip1", "line": 1, "active_low": True},
        # AO 判定：实测 raw 最灵敏（本底 ~100，吹烟冲到 ~158，DO 一直不动），
        "ao": {"enable": True, "alarm": False, "alarm_raw": 150,
               "alarm_hysteresis": 10, "alarm_volt": None,
               "iio_path": "/sys/bus/iio/devices/iio:device0/in_voltage0_raw",
               "vref": 1.8, "bits": 10, "divider": 0.4},
        "poll_sec": 0.5, "debounce": 3, "cooldown": 120, "value_interval": 10,
        "log_interval": 10,   # 日志里每几秒打一行读数（0=只在状态变化/告警时打）
    },
}


def _deep_update(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _deep_update(cfg, data)
        except Exception as e:
            print("WARN: 无法解析配置 %s: %s" % (path, e))
    sm = cfg.setdefault("smoke", {})
    do = sm.setdefault("do", {})
    if "gpiochip" in sm:
        do.setdefault("gpiochip", sm.pop("gpiochip"))
    if "line" in sm:
        do.setdefault("line", sm.pop("line"))
    return cfg


# ---------------- 读取 ----------------
def read_do(chip, line):
    """用 gpioget 读 DO 原始电平，返回 '1'/'0'，失败返回 None。"""
    try:
        out = subprocess.check_output(["gpioget", chip, str(line)],
                                      stderr=subprocess.DEVNULL,
                                      universal_newlines=True)
        return out.strip().split()[-1]
    except Exception:
        return None


def read_ao(path, vref=1.8, bits=10):
    """读 ADC 原始值，返回 (raw, adc_volt)；读不到返回 (None, None)。"""
    try:
        with open(path) as f:
            raw = int(f.read().strip())
    except Exception:
        return None, None
    full = float((1 << int(bits)) - 1)
    return raw, raw / full * float(vref)


def do_is_gas(raw, active_low=True):
    """把 DO 原始电平翻译成"是否检测到烟雾"。"""
    if raw is None:
        return None
    return (raw == "0") if active_low else (raw == "1")


def publish(client, topic, payload, qos=1, retain=False):
    try:
        client.publish(topic, json.dumps(payload, ensure_ascii=False),
                       qos=qos, retain=retain)
    except Exception as e:
        print("WARN: MQTT 发布失败: %s" % e)


def sensor_volts(adc_volt, divider):
    """把 ADC 端电压还原成传感器 AO 端电压（分压比 divider = 2/5 = 0.4）。"""
    if adc_volt is None or not divider:
        return None
    return adc_volt / float(divider)


def main():
    ap = argparse.ArgumentParser(description="MQ-2 烟雾传感器 MQTT 发布器")
    ap.add_argument("--config", default=pick_config(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--gpiochip", default=None, help="覆盖 DO 的 gpiochip（默认 gpiochip1）")
    ap.add_argument("--line", type=int, default=None, help="覆盖 DO 的 line（默认 1）")
    ap.add_argument("--active-low", dest="active_low", action="store_true", default=None,
                    help="低电平=检测到烟雾（MQ-2 常见接法）")
    ap.add_argument("--active-high", dest="active_low", action="store_false",
                    help="高电平=检测到烟雾")
    ap.add_argument("--broker", default=None)
    ap.add_argument("--camera-id", default=None)
    ap.add_argument("--watch", type=int, default=0,
                    help="只做实时读数自检：打印 N 秒的 DO 电平与 AO 电压，不连 MQTT")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sm = cfg.setdefault("smoke", {})
    do = sm.setdefault("do", {})
    ao = sm.setdefault("ao", {})
    mq = cfg.setdefault("mqtt", {})

    if args.gpiochip:
        do["gpiochip"] = args.gpiochip
    if args.line is not None:
        do["line"] = args.line
    if args.active_low is not None:
        do["active_low"] = args.active_low
    if args.broker:
        mq["broker"] = args.broker
    if args.camera_id:
        cfg["camera_id"] = args.camera_id

    chip = do.get("gpiochip", "gpiochip1")
    line = int(do.get("line", 1))
    active_low = bool(do.get("active_low", True))
    poll = float(sm.get("poll_sec", 0.5))

    # ---------- --watch：接线/极性自检（不连 MQTT） ----------
    if args.watch:
        print("实时读数 %d 秒：DO=%s line %d（active_low=%s），AO=%s"
              % (args.watch, chip, line, active_low, ao.get("iio_path")))
        print("  对着传感器吹一口烟气（或用打火机放气），观察 DO 是否翻转、AO 电压是否上升")
        t0 = time.time()
        raws = []          # 累计本次读到的 raw，结束时给标定参考
        while time.time() - t0 < args.watch:
            v = read_do(chip, line)
            raw, av = read_ao(ao.get("iio_path"), ao.get("vref", 1.8), ao.get("bits", 10))
            sv = sensor_volts(av, ao.get("divider", 0.4))
            if raw is not None:
                raws.append(raw)
            do_gas = do_is_gas(v, active_low)
            thr = ao.get("alarm_raw")
            ao_gas = (raw is not None and thr is not None and raw >= float(thr))
            gas = bool(do_gas) or ao_gas
            print("  DO=%-4s AO raw=%-5s%s ADC=%-6s 传感器=%s V -> %s%s"
                  % (v, raw if raw is not None else "-",
                     ("(阈值%s)" % int(thr)) if thr is not None else "",
                     ("%.3f" % av) if av is not None else "-",
                     ("%.2f" % sv) if sv is not None else "-",
                     ("烟雾!" if gas else "正常") if v is not None or raw is not None else "读取失败",
                     " [AO]" if ao_gas and not do_gas else (" [DO]" if do_gas else "")))
            time.sleep(poll)
        if raws:
            lo, hi = min(raws), max(raws)
            avg = sum(raws) / float(len(raws))
            print("本次 %d 秒：raw 最小 %d / 最大 %d / 平均 %.1f（峰值增量 +%d）"
                  % (args.watch, lo, hi, avg, hi - lo))
            if hi - lo >= 20:
                print("  → 传感器响应正常。标定建议：阈值取 最小+(峰增量×0.6~0.7) ≈ %d"
                      % (lo + int((hi - lo) * 0.65)))
            else:
                print("  → 期间几乎没有变化：确认气体有没有进到传感器网罩（贴近、朝下吹），"
                      "以及传感器是否刚上电/刚被测过（需要恢复时间）")

        if read_do(chip, line) is None:
            print("提示：DO 读不到值 —— 确认已装 gpiod（sudo apt install gpiod），"
                  "并用 gpiodetect / gpioinfo %s 核对 chip 与 line 号" % chip)
        return 0

    # ---------- 正式发布 ----------
    broker = str(mq.get("broker") or "").strip()
    if not broker:
        print("ERROR: 未配置 MQTT broker（用 --broker 或配置 mqtt.broker）")
        return 2
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("ERROR: 当前用户没装 paho-mqtt。")
        print("       注意服务以 root 运行时，用 `pip3 install --user` 装的包对 root 不可见：")
        print("         方案1) sudo apt install python3-paho-mqtt   或   sudo pip3 install paho-mqtt")
        print("         方案2) 把服务的 User= 改回 firefly，并用 gpio 组权限读 /dev/gpiochip*")
        return 2

    client = mqtt.Client(client_id="smoke-%d" % os.getpid())
    if mq.get("username"):
        client.username_pw_set(mq.get("username"), mq.get("password", ""))
    try:
        client.connect(broker, int(mq.get("port", 1883)), keepalive=60)
        client.loop_start()
    except Exception as e:
        print("ERROR: MQTT 连接 %s 失败: %s" % (broker, e))
        return 2

    debounce = int(sm.get("debounce", 3))
    cooldown = float(sm.get("cooldown", 120))
    value_int = float(sm.get("value_interval", 10))
    log_int = float(sm.get("log_interval", 10))
    hb_int = float(mq.get("heartbeat_interval", 30))
    prefix = sm.get("topic_prefix") or mq.get("topic_prefix", "safe")
    cid = cfg.get("camera_id", "cam1")

    t_alert = "%s/%s/alert" % (prefix, cid)
    t_value = "%s/%s/smoke/value" % (prefix, cid)
    t_hb = "%s/%s/smoke/heartbeat" % (prefix, cid)

    print("MQ-2 启动: DO=%s line %d (active_low=%s) | AO=%s | MQTT %s"
          % (chip, line, active_low, ao.get("iio_path"), broker))
    print("topics: %s / %s / %s" % (t_alert, t_value, t_hb))

    gas_count = 0
    last_alert = 0.0
    last_value = 0.0
    last_hb = 0.0
    last_log = 0.0
    prev_do = None
    prev_gas = None
    ao_latched = False
    t0 = time.time()

    try:
        while True:
            now = time.time()
            raw_do = read_do(chip, line)
            raw_ao, adc_v = read_ao(ao.get("iio_path"), ao.get("vref", 1.8),
                                    ao.get("bits", 10))
            sv = sensor_volts(adc_v, ao.get("divider", 0.4))

            if raw_do is None:
                # DO 读不到（没装 gpiod / 权限不足 / chip,line 不对）：
                # 不能静默退出循环，否则 MQTT 上看不出"服务活着但读不到传感器"。
                # 照常发遥测（标 error）和心跳，只是不参与报警判定。
                if now - last_value >= value_int:
                    last_value = now
                    bad = {"event": "value", "camera_id": cid,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "do": None, "gas": None,
                           "error": "gpioget failed: %s line %d (装 gpiod? 权限? chip/line?)"
                                    % (chip, line)}
                    if raw_ao is not None:
                        bad.update({"ao_raw": raw_ao, "adc_volt": round(adc_v, 4),
                                    "ao_volt": round(sv, 3)})
                    publish(client, t_value, bad, retain=True)
                if now - last_hb >= hb_int:
                    last_hb = now
                    publish(client, t_hb, {
                        "event": "heartbeat", "camera_id": cid,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "uptime": round(now - t0, 1), "alive": True,
                        "do_ok": False,
                    }, retain=True)
                if log_int > 0 and (last_log == 0.0 or now - last_log >= log_int):
                    last_log = now
                    print("WARN: gpioget 失败（gpiod 没装? 权限不足? chip/line 不对?）"
                          "——遥测已发并标注 error")
                time.sleep(2)
                continue

            # ---- 判定：DO（若模块电位器调好会翻转）或 AO raw（实测最灵敏）----
            thr_raw = ao.get("alarm_raw")
            ao_alarm = False
            if ao.get("alarm"):
                if thr_raw is not None and raw_ao is not None:
                    hy = float(ao.get("alarm_hysteresis", 10))
                    if ao_latched:                      # 已触发：要跌破"阈值-迟滞"才解除
                        ao_latched = raw_ao > (float(thr_raw) - hy)
                    else:                               # 未触发：达到阈值才触发
                        ao_latched = raw_ao >= float(thr_raw)
                    ao_alarm = ao_latched
                elif ao.get("alarm_volt") is not None and sv is not None:
                    ao_alarm = sv >= float(ao["alarm_volt"])
            gas = bool(do_is_gas(raw_do, active_low)) or ao_alarm

            # ---- 日志 ----
            # 判定翻转立刻打一行（最有诊断价值）；数值按 log_interval 周期打，避免刷爆 journal
            if prev_gas is not None and gas != prev_gas:
                print("状态变化: %s -> %s（DO=%s AO raw=%s 传感器=%sV）"
                      % ("烟雾" if prev_gas else "正常",
                         "烟雾" if gas else "正常", raw_do,
                         raw_ao if raw_ao is not None else "-",
                         ("%.2f" % sv) if sv is not None else "-"))
            prev_gas = gas
            prev_do = raw_do

            gas_count = gas_count + 1 if gas else 0

            if log_int > 0 and (last_log == 0.0 or now - last_log >= log_int):
                last_log = now
                print("DO=%s %s | AO raw=%s%s adc=%sV 传感器=%sV | 连续确认 %d/%d"
                      % (raw_do, "烟雾!" if gas else "正常",
                         raw_ao if raw_ao is not None else "-",
                         (" (阈值%s)" % int(thr_raw)) if thr_raw is not None else "",
                         ("%.3f" % adc_v) if adc_v is not None else "-",
                         ("%.2f" % sv) if sv is not None else "-",
                         gas_count, debounce))

            if gas_count >= debounce and now - last_alert >= cooldown:
                last_alert = now
                payload = {
                    "event": "smoke", "camera_id": cid,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "level": "high", "do": raw_do, "ao_raw": raw_ao,
                }
                if sv is not None:
                    payload["ao_volt"] = round(sv, 3)
                publish(client, t_alert, payload)
                print("!!! 检测到烟雾 -> %s (DO=%s, AO raw=%s, 传感器=%sV)"
                      % (t_alert, raw_do, raw_ao,
                         ("%.2f" % sv) if sv is not None else "-"))

            if now - last_value >= value_int:
                last_value = now
                payload = {"event": "value", "camera_id": cid,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "do": raw_do, "gas": gas}
                if raw_ao is not None:
                    payload["ao_raw"] = raw_ao
                    payload["adc_volt"] = round(adc_v, 4)
                    payload["ao_volt"] = round(sv, 3)
                publish(client, t_value, payload, retain=True)

            if now - last_hb >= hb_int:
                last_hb = now
                publish(client, t_hb, {
                    "event": "heartbeat", "camera_id": cid,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uptime": round(now - t0, 1), "alive": True, "do_ok": True,
                }, retain=True)

            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())