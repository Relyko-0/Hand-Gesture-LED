"""
Govee lights for Hand Led.py.

Tries the LAN API first (fast, about 10 updates a second). If no lights answer on
the LAN, it falls back to the cloud API with your key (slower, about 1 update a
second, because Govee rate-limits the cloud API).

API key: set the GOVEE_API_KEY environment variable, or paste the key into
govee_key.txt next to this file.
LAN: turn on "LAN Control" for each light in the Govee Home app (light settings).
"""

import json
import os
import socket
import threading
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, "govee_key.txt")

LAN_SCAN_ADDR = ("239.255.255.250", 4001)   # lights listen for scans here
LAN_REPLY_PORT = 4002                       # ...and answer to this port
LAN_CMD_PORT = 4003                         # commands go to each light's IP on this port
CLOUD_URL = "https://openapi.api.govee.com/router/api/v1"

LAN_INTERVAL = 0.1     # seconds between color updates on the LAN
CLOUD_INTERVAL = 1.0   # seconds between color updates over the cloud (rate limited)

# Which lights to control, by name (as in the Govee app) or model number.
# The model number is what's used on the LAN. Leave empty to control every light.
ONLY_LIGHTS = ["Strip Light", "H612F"]


# The cloud only manages ~1 update a second, so by default keep looking on the LAN
# instead of falling back to it. Set True to use the cloud when the LAN finds nothing.
USE_CLOUD_FALLBACK = False


def wanted(*labels):
    return not ONLY_LIGHTS or any(label in ONLY_LIGHTS for label in labels)


def load_key():
    key = os.environ.get("GOVEE_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    return key or None


# ----------------------------------------------------------------- LAN API

def lan_scan(timeout=3.0):
    """Ask every Govee light on the network to say hello. Returns {ip: sku}."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", LAN_REPLY_PORT))
    except OSError:
        sock.close()
        return {}
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.settimeout(0.2)
    scan = {"msg": {"cmd": "scan", "data": {"account_topic": "reserve"}}}
    found = {}
    try:
        end = time.monotonic() + timeout
        next_scan = 0
        while time.monotonic() < end:
            if time.monotonic() >= next_scan:   # repeat the scan; lights sometimes miss one
                sock.sendto(json.dumps(scan).encode(), LAN_SCAN_ADDR)
                next_scan = time.monotonic() + 0.5
            try:
                data, _ = sock.recvfrom(4096)
                info = json.loads(data)["msg"]["data"]
                found[info["ip"]] = info.get("sku", "?")
            except socket.timeout:
                continue
            except (ValueError, KeyError, TypeError):
                continue   # not a scan reply
    except OSError:
        pass
    finally:
        sock.close()
    return found


def lan_send(sock, ip, cmd, data):
    sock.sendto(json.dumps({"msg": {"cmd": cmd, "data": data}}).encode(), (ip, LAN_CMD_PORT))


# ----------------------------------------------------------------- cloud API

def cloud_request(key, method, path, body=None):
    req = urllib.request.Request(
        CLOUD_URL + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Govee-API-Key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def cloud_devices(key):
    """Lights on the account that can show an RGB color: [(sku, device_id, name)]."""
    devices = cloud_request(key, "GET", "/user/devices").get("data", [])
    return [(d["sku"], d["device"], d.get("deviceName") or d["sku"]) for d in devices
            if any(c.get("instance") == "colorRgb" for c in d.get("capabilities", []))]


def cloud_control(key, sku, device, cap_type, instance, value):
    reply = cloud_request(key, "POST", "/device/control", {
        "requestId": str(uuid.uuid4()),
        "payload": {"sku": sku, "device": device,
                    "capability": {"type": cap_type, "instance": instance, "value": value}}})
    if reply.get("code") != 200:
        raise RuntimeError(reply.get("msg") or reply.get("message") or reply)


# ----------------------------------------------------------------- the lights

class GoveeLights:
    """Call set_color((r, g, b)) as often as you like; a background thread sends
    only the latest color, only when it changed, at a rate the lights can take.
    The lights switch to start_color as soon as they connect."""

    def __init__(self, start_color=(255, 0, 0)):
        self.mode = None            # "LAN", "cloud", or None
        self.names = []
        self.state = "connecting..."
        self._want = tuple(start_color)
        self._sent = None
        self._last_send = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_color(self, rgb):
        with self._lock:
            self._want = tuple(int(v) for v in rgb)

    def status(self):
        return f"Govee: {self.state}"

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _connect(self):
        while not self._stop.is_set():
            lan = {ip: sku for ip, sku in lan_scan().items() if wanted(sku)}
            if lan:
                self.mode, self._ips, self.names = "LAN", list(lan), list(lan.values())
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                return LAN_INTERVAL
            if USE_CLOUD_FALLBACK:
                break
            self.state = "searching for lights on Wi-Fi..."
        else:
            return None
        key = load_key()
        if not key:
            self.state = "no lights on LAN, no API key"
            return None
        try:
            self._devices = [d for d in cloud_devices(key) if wanted(d[0], d[2])]
        except Exception as e:
            self.state = f"cloud error ({e})"
            return None
        if not self._devices:
            self.state = "no color lights on account"
            return None
        self.mode, self._key, self.names = "cloud", key, [n for _, _, n in self._devices]
        return CLOUD_INTERVAL

    def _send(self, cmd):
        """cmd is ("on",) or ("color", (r, g, b))."""
        if self.mode == "LAN":
            for ip in self._ips:
                if cmd[0] == "on":
                    lan_send(self._sock, ip, "turn", {"value": 1})
                else:
                    r, g, b = cmd[1]
                    lan_send(self._sock, ip, "colorwc",
                             {"color": {"r": r, "g": g, "b": b}, "colorTemInKelvin": 0})
        else:
            failed = []
            for sku, device, name in self._devices:
                try:
                    if cmd[0] == "on":
                        cloud_control(self._key, sku, device,
                                      "devices.capabilities.on_off", "powerSwitch", 1)
                    else:
                        r, g, b = cmd[1]
                        cloud_control(self._key, sku, device, "devices.capabilities.color_setting",
                                      "colorRgb", (r << 16) | (g << 8) | b)
                except Exception as e:
                    failed.append(f"{name}: {e}")
            # one offline light shouldn't stop the others; just report it
            if failed and failed != getattr(self, "_last_failed", None):
                print("Govee: " + "; ".join(failed))
            self._last_failed = failed

    def _run(self):
        interval = self._connect()
        if interval is None:
            print(self.status())
            return
        self.state = f"{self.mode}, {len(self.names)} light(s)"
        print(f"{self.status()}: {', '.join(self.names)}")
        try:
            self._send(("on",))
        except Exception as e:
            print(f"Govee: couldn't turn lights on ({e})")
        self._stop.wait(0.5)   # give the lights a moment to power up before the first color

        while not self._stop.wait(interval):
            with self._lock:
                want = self._want
            # LAN messages can get lost, so resend the current color every second or so
            resend = self.mode == "LAN" and time.monotonic() - self._last_send > 1.0
            if want is None or (want == self._sent and not resend):
                continue
            try:
                self._send(("color", want))
                self._sent, self._last_send = want, time.monotonic()
                self.state = f"{self.mode}, {len(self.names)} light(s)"
            except Exception as e:
                self.state = f"{self.mode} error, retrying"
                print(f"Govee: send failed ({e})")
                self._stop.wait(5)   # back off (e.g. cloud rate limit)


if __name__ == "__main__":
    # Quick test: python govee_lights.py  -> lights go red, green, blue
    lights = GoveeLights()
    while lights.mode is None and lights._thread.is_alive():
        time.sleep(0.1)
    if lights.mode:
        for rgb in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
            print("color", rgb)
            lights.set_color(rgb)
            time.sleep(2)
    lights.close()
