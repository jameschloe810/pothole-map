# integrate_gps_standalone.py
from pathlib import Path
import time, json, os, select, termios

AT_PORT = "/dev/ttyUSB2"  # change if AT port moves

def _open_serial(path, baud=115200, timeout_s=0.2):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)  # [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
    attrs[0] = termios.IGNPAR   # iflag
    attrs[1] = 0                # oflag
    attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8  # cflag
    attrs[3] = 0                # lflag
    speed = getattr(termios, f'B{baud}', termios.B115200)
    attrs[4] = speed; attrs[5] = speed
    cc = attrs[6]; cc[termios.VMIN] = 0; cc[termios.VTIME] = max(1, int(timeout_s*10)); attrs[6] = cc
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd

def _write(fd, s): os.write(fd, (s + "\r\n").encode())

def _read_all(fd, max_wait=0.5):
    chunks=[]; end=time.time()+max_wait
    while time.time() < end:
        r,_,_ = select.select([fd], [], [], 0.1)
        if not r: continue
        try: data = os.read(fd, 4096)
        except BlockingIOError: data = b""
        if data: chunks.append(data)
    return b"".join(chunks).decode(errors="ignore")

def at_cmd(fd, cmd, wait=0.3):
    _ = _read_all(fd, 0.1)
    _write(fd, cmd)
    time.sleep(wait)
    return _read_all(fd, 0.4)

def _dm_to_dd(dm, is_lon=False):
    if not dm or dm in ("0","0.0"): return None
    head = dm.split(".")[0]
    deg_digits = 3 if (is_lon and len(head) >= 5) else 2
    deg = int(dm[:deg_digits]); mins = float(dm[deg_digits:])
    return deg + mins/60.0

def get_fix(fd, retries=12, pause=0.25):
    for _ in range(retries):
        out = at_cmd(fd, "AT+CGPSINFO")
        if "+CGPSINFO:" not in out:
            time.sleep(pause); continue
        line = out.split("+CGPSINFO:")[-1].strip().splitlines()[0].strip()
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8 or not p[0] or not p[2]:
            time.sleep(pause); continue
        lat = _dm_to_dd(p[0]); lon = _dm_to_dd(p[2], is_lon=True)
        if lat is None or lon is None:
            time.sleep(pause); continue
        if p[1] == "S": lat *= -1
        if p[3] == "W": lon *= -1
        alt = float(p[6]) if p[6] else None
        spd = float(p[7]) if p[7] else None
        return {"lat": lat, "lon": lon, "alt_m": alt, "speed_kn": spd, "utc_date": p[4], "utc_time": p[5]}
    return {}

# ---- one-time init at program start ----
out_dir = Path("/home/pi/pothole"); out_dir.mkdir(parents=True, exist_ok=True)
fd = _open_serial(AT_PORT, 115200)
at_cmd(fd, "ATE0")
if "1" not in at_cmd(fd, "AT+CGPS?"):
    at_cmd(fd, "AT+CGPS=1")
    time.sleep(1.0)

def log_detection(image_path, conf, cls_="pothole"):
    fix = get_fix(fd, retries=12, pause=0.25)
    rec = {"ts": time.time(), "image": image_path, "conf": float(conf), "cls": cls_, "gps": fix or None}
    (out_dir/"latest_detection.json").write_text(json.dumps(rec))
    with (out_dir/"detections.csv").open("a") as f:
        f.write("{ts},{lat},{lon},{alt},{spd},{img},{conf},{cls}\n".format(
            ts=rec["ts"],
            lat=(fix.get("lat") if fix else ""),
            lon=(fix.get("lon") if fix else ""),
            alt=(fix.get("alt_m") if fix else ""),
            spd=(fix.get("speed_kn") if fix else ""),
            img=image_path, conf=conf, cls=cls_))
    print("Logged:", rec)
