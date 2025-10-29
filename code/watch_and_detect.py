#!/usr/bin/env python3
import os, time, csv, glob, shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
from datetime import datetime

# -----------------------------
# Config (env overrides)
# -----------------------------
WATCH_DIR   = os.path.expanduser(os.environ.get("WATCH_DIR", "~/captures"))
OUT_DIR     = os.path.expanduser(os.environ.get("OUT_DIR",   "~/detections"))
MODEL_PATH  = os.path.expanduser(os.environ.get("MODEL_PATH","~/best.onnx"))

CONF_THR    = float(os.environ.get("CONF_THR",  "0.55"))
IOU_THR     = float(os.environ.get("IOU_THR",   "0.80"))
MIN_AREA    = float(os.environ.get("MIN_AREA",  "800"))
PRE_TOPK    = int(os.environ.get("PRE_TOPK",    "6000"))
MAX_DET     = int(os.environ.get("MAX_DET",     "8"))
POLL_SEC    = float(os.environ.get("POLL_SEC",  "0.5"))

# File stability / staging (prevents racing with the camera)
STABLE_CHECKS   = int(os.environ.get("STABLE_CHECKS", "3"))      # consecutive equal stats
STABLE_INTERVAL = float(os.environ.get("STABLE_INTERVAL", "0.2"))# seconds between checks
STAGE_DIR       = os.path.join(OUT_DIR, "_stage")

# FPS logging
FPS_EMA_ALPHA      = float(os.environ.get("FPS_EMA_ALPHA", "0.2"))
FPS_REPORT_EVERY   = int(os.environ.get("FPS_REPORT_EVERY", "10"))

# CSV (unchanged schema)
DET_CSV     = os.path.join(OUT_DIR, "detections.csv")
DET_FIELDS  = ['image','conf','x1','y1','x2','y2','w','h','area_px','area_norm','lat','lon','alt','spd']

# -----------------------------
# Optional GPS integration
# -----------------------------
gps_available = False
gps_get_fix = None
try:
    import integrate_gps_standalone as gpsmod  # your AT helper
    if hasattr(gpsmod, "get_fix"):
        gps_available = True
        gps_get_fix = gpsmod.get_fix
except Exception:
    gps_available = False
    gps_get_fix = None

# -----------------------------
# Utils
# -----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    denom = area_a + area_b - inter
    return inter/denom if denom>0 else 0.0

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_det:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        idxs = rest[ious <= iou_thr]
    return keep

def draw_boxes_with_conf(img: Image.Image, boxes_xyxy: np.ndarray, scores: np.ndarray, color=(255, 0, 0)) -> Image.Image:
    """Draw rectangles + small confidence label on top-left."""
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{s:.2f}"
        tw = draw.textlength(label); th = 12
        bx1 = x1; by1 = max(0, y1 - th - 4); bx2 = x1 + tw + 6; by2 = y1
        draw.rectangle([bx1, by1, bx2, by2], fill=color)
        draw.text((x1 + 3, by1 + 1), label, fill=(255,255,255))
    return img

def clamp_and_log(image_name_for_csv, conf, x1, y1, x2, y2, w, h, lat=None, lon=None, alt=None, spd=None, writer=None) -> bool:
    x1 = max(0.0, min(float(x1), float(w))); y1 = max(0.0, min(float(y1), float(h)))
    x2 = max(0.0, min(float(x2), float(w))); y2 = max(0.0, min(float(y2), float(h)))
    if x2 <= x1 or y2 <= y1: return False
    area_px = (x2 - x1) * (y2 - y1)
    if area_px < MIN_AREA: return False
    area_norm = (area_px / (float(w) * float(h))) if (w and h) else ""
    row = {
        'image': image_name_for_csv,
        'conf': f'{float(conf):.4f}',
        'x1': f'{x1:.1f}', 'y1': f'{y1:.1f}',
        'x2': f'{x2:.1f}', 'y2': f'{y2:.1f}',
        'w': f'{int(w)}', 'h': f'{int(h)}',
        'area_px': f'{area_px:.1f}',
        'area_norm': f'{area_norm:.6f}' if area_norm != "" else '',
        'lat': '' if lat is None else f'{float(lat)}',
        'lon': '' if lon is None else f'{float(lon)}',
        'alt': '' if alt is None else f'{float(alt)}',
        'spd': '' if spd is None else f'{float(spd)}',
    }
    if writer: writer.writerow(row)
    return True

def load_session(model_path: str):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    in_name = inp.name
    shape = list(inp.shape)
    for i,v in enumerate(shape):
        if not isinstance(v, int):
            shape[i] = int(v) if v is not None else -1
    nhwc = False
    if len(shape) == 4 and shape[1] == 3:
        H, W = int(shape[2]), int(shape[3])
    elif len(shape) == 4 and shape[3] == 3:
        nhwc = True; H, W = int(shape[1]), int(shape[2])
    else:
        raise RuntimeError(f"Unexpected input shape {inp.shape}")
    out_names = [o.name for o in sess.get_outputs()]
    return sess, in_name, (H, W), nhwc, out_names

def preprocess(image_path: str, H: int, W: int, nhwc: bool):
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    img_r = img.resize((W, H), Image.BILINEAR)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0
    x = arr.reshape(1, H, W, 3) if nhwc else np.transpose(arr, (2,0,1)).reshape(1, 3, H, W)
    return img, x, src_w, src_h

def decode_outputs(outs, H: int, W: int, conf_thr: float, pre_topk: int) -> Tuple[np.ndarray, np.ndarray]:
    det = None
    for y in outs:
        if isinstance(y, np.ndarray) and y.ndim == 3 and y.shape[0] == 1 and (y.shape[1] >= 6):
            det = y; break
    if det is None: return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    det = det[0]; C, N = det.shape; det = det.T
    x = det[:,0]; y = det[:,1]; w = det[:,2]; h = det[:,3]
    obj = sigmoid(det[:,4]) if C >= 5 else np.ones((N,), dtype=np.float32)
    conf = obj.copy()
    if C > 5:
        cls_scores = sigmoid(det[:,5:]); max_cls = cls_scores.max(axis=1); conf = conf * max_cls
    keep = conf >= conf_thr
    if not np.any(keep): return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    x = x[keep]; y = y[keep]; w = w[keep]; h = h[keep]; conf = conf[keep]
    x1 = x - w/2.0; y1 = y - h/2.0; x2 = x + w/2.0; y2 = y + h/2.0
    boxes = np.stack([x1,y1,x2,y2], axis=1).astype(np.float32)
    if pre_topk and boxes.shape[0] > pre_topk:
        idx = np.argsort(conf)[-pre_topk:]; boxes = boxes[idx]; conf = conf[idx]
    return boxes, conf

# -----------------------------
# File stability + staging
# -----------------------------
def stat_meta(path: str):
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)

def wait_until_stable(path: str, checks: int, interval: float) -> bool:
    try:
        last = stat_meta(path)
        stable = 1
        while stable < checks:
            time.sleep(interval)
            cur = stat_meta(path)
            if cur == last:
                stable += 1
            else:
                last = cur
                stable = 1
        return True
    except FileNotFoundError:
        return False

def stage_copy(src_path: str, stage_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(src_path))[0]
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    staged = os.path.join(stage_dir, f"{stem}_{ts}_src.jpg")
    shutil.copy2(src_path, staged)
    return staged

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(WATCH_DIR, exist_ok=True)
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(STAGE_DIR, exist_ok=True)

    # CSV writer
    need_header = (not os.path.exists(DET_CSV)) or (os.path.getsize(DET_CSV) == 0)
    det_fh = open(DET_CSV, 'a', newline='')
    det_writer = csv.DictWriter(det_fh, fieldnames=DET_FIELDS)
    if need_header: det_writer.writeheader()

    # Model
    print(f"Loading model: {MODEL_PATH}")
    sess, in_name, (H, W), nhwc, out_names = load_session(MODEL_PATH)
    print(f"Input: {('NHWC' if nhwc else 'NCHW')} {H}x{W}")
    print(f"Watching: {WATCH_DIR}")
    if gps_available: print("GPS detected; will log fixes when available.")

    processed_meta = {}  # path -> (mtime_ns, size)
    ema_fps = None; proc_count = 0; last_report = time.perf_counter()

    try:
        while True:
            paths = sorted(glob.glob(os.path.join(WATCH_DIR, "*.jpg")))
            if not paths:
                time.sleep(POLL_SEC); continue

            for src in paths:
                try:
                    meta = stat_meta(src)
                except FileNotFoundError:
                    continue

                # Skip if this exact content was already processed
                if processed_meta.get(src) == meta:
                    continue

                # Wait for the camera to finish writing
                if not wait_until_stable(src, STABLE_CHECKS, STABLE_INTERVAL):
                    continue

                # Confirm again (might have changed while we waited)
                try:
                    meta2 = stat_meta(src)
                except FileNotFoundError:
                    continue
                if processed_meta.get(src) == meta2:
                    continue

                # Snapshot content to a staged copy
                staged = stage_copy(src, STAGE_DIR)

                # Prepare unique annotated filename (also written into CSV)
                src_stem = os.path.splitext(os.path.basename(src))[0]
                ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                annot_name = f"{src_stem}_{ts}_annot.jpg"
                out_path = os.path.join(OUT_DIR, annot_name)

                t0 = time.perf_counter()
                try:
                    # Preprocess FROM STAGED COPY
                    src_img, x, src_w, src_h = preprocess(staged, H, W, nhwc)
                    outs = sess.run(out_names, {in_name: x})
                    boxes_in, conf_in = decode_outputs(outs, H, W, CONF_THR, PRE_TOPK)

                    if boxes_in.shape[0] > 0:
                        keep = nms(boxes_in, conf_in, IOU_THR, MAX_DET)
                        boxes = boxes_in[keep]; confs = conf_in[keep]

                        # Rescale to original size
                        scale_x = float(src_w) / float(W); scale_y = float(src_h) / float(H)
                        boxes_xyxy = boxes.copy()
                        boxes_xyxy[:, [0,2]] *= scale_x
                        boxes_xyxy[:, [1,3]] *= scale_y

                        # GPS (best-effort)
                        lat = lon = alt = spd = None
                        if gps_available and gps_get_fix is not None:
                            try:
                                fix = gps_get_fix(getattr(gpsmod, 'fd', None) or None) if hasattr(gpsmod, 'fd') else gps_get_fix(None)
                                if isinstance(fix, dict) and fix.get('lat') is not None and fix.get('lon') is not None:
                                    lat = fix.get('lat'); lon = fix.get('lon')
                                    alt = fix.get('alt_m'); spd = fix.get('speed_kn')
                            except Exception:
                                pass

                        # Draw + CSV
                        img_draw = src_img.copy()
                        valid = 0
                        for (x1,y1,x2,y2), c in zip(boxes_xyxy, confs):
                            if clamp_and_log(
                                image_name_for_csv=annot_name,
                                conf=c, x1=x1, y1=y1, x2=x2, y2=y2,
                                w=src_w, h=src_h, lat=lat, lon=lon, alt=alt, spd=spd,
                                writer=det_writer
                            ):
                                valid += 1
                        if valid:
                            img_draw = draw_boxes_with_conf(img_draw, boxes_xyxy, confs, color=(255,0,0))
                            img_draw.save(out_path, quality=92)
                            print(f"{os.path.basename(src)} -> kept {valid} box(es); wrote {annot_name}")
                        else:
                            print(f"{os.path.basename(src)} -> no valid boxes after clamping/area")
                        det_fh.flush()
                    else:
                        print(f"{os.path.basename(src)} -> no detections")

                except Exception as e:
                    print("Error on", src, "->", e)

                finally:
                    processed_meta[src] = meta2  # mark this (mtime,size) as processed
                    try: os.remove(staged)      # clean up snapshot
                    except Exception: pass

                    # FPS
                    dt = time.perf_counter() - t0
                    inst_fps = 1.0/dt if dt>0 else float('inf')
                    ema_fps = inst_fps if ema_fps is None else (FPS_EMA_ALPHA*inst_fps + (1-FPS_EMA_ALPHA)*ema_fps)
                    proc_count += 1
                    if proc_count % FPS_REPORT_EVERY == 0:
                        now = time.perf_counter()
                        window_fps = FPS_REPORT_EVERY / (now - last_report)
                        last_report = now
                        print(f"[THROUGHPUT] last {FPS_REPORT_EVERY}: {window_fps:.2f} fps | inst {inst_fps:.2f} | ema {ema_fps:.2f} | {dt*1000:.1f} ms")

            time.sleep(POLL_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        try: det_fh.close()
        except Exception: pass

if __name__ == "__main__":
    main()
