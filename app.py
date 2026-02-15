#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import threading
import queue
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, send_from_directory

# ─── DIRECTORIES ──────────────────────────────────────────────────────
CAPTURES_DIR = Path("captures")
HEATMAP_DIR  = Path("heatmap_images")
CAPTURES_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# ─── YOLO ─────────────────────────────────────────────────────────────
print("[YOLO] Loading model...")
from ultralytics import YOLO
import numpy as np
_MODEL = YOLO("yolov8n.pt")
_MODEL(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
print("[YOLO] Model ready.")

# ─── PI CAMERA (Picamera2) ────────────────────────────────────────────
from picamera2 import Picamera2
_picam = Picamera2()
_picam.configure(_picam.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
_picam.start()
print("[PICAM] Picamera2 started.")

# ─── THERMAL (MLX90640) ───────────────────────────────────────────────
try:
    import board, busio, adafruit_mlx90640
    _i2c = busio.I2C(board.SCL, board.SDA)
    _MLX = adafruit_mlx90640.MLX90640(_i2c)
    _MLX.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    _THERMAL_OK = True
    print("[THERMAL] MLX90640 ready.")
except Exception as e:
    _MLX = None
    _THERMAL_OK = False
    print(f"[THERMAL] Not available: {e}")

# ─── GPS (NEO-6M serial) ──────────────────────────────────────────────
GPS_PORT = "/dev/ttyUSB0"
GPS_BAUD = 9600

try:
    import serial as _serial
    _GPS_SER = _serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    _GPS_OK  = True
    print(f"[GPS] Serial open on {GPS_PORT}")
except Exception as e:
    _GPS_SER = None
    _GPS_OK  = False
    print(f"[GPS] Not available: {e}")

# ─── DROPPER (servo) ──────────────────────────────────────────────────
try:
    import dropper as _dropper
    _dropper.initialize_servos()
    _DROPPER_OK = True
    print("[DROPPER] Servos ready.")
except Exception as e:
    _dropper = None
    _DROPPER_OK = False
    print(f"[DROPPER] Not available: {e}")

# ─── CONFIG ───────────────────────────────────────────────────────────
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
LOCK_ON_SECONDS = 3.0
CONFIDENCE_MIN  = 0.45

BOX_W  = int(FRAME_WIDTH  * 0.35)
BOX_H  = int(FRAME_HEIGHT * 0.45)
BOX_X1 = (FRAME_WIDTH  - BOX_W) // 2
BOX_Y1 = (FRAME_HEIGHT - BOX_H) // 2
BOX_X2 = BOX_X1 + BOX_W
BOX_Y2 = BOX_Y1 + BOX_H

app = Flask(__name__)


# ─── THERMAL HELPER ───────────────────────────────────────────────────
def _read_thermal():
    if not _THERMAL_OK or _MLX is None:
        return None, None, None
    buf = np.zeros((24 * 32,), dtype=float)
    for attempt in range(5):
        try:
            _MLX.getFrame(buf)
            data = np.fliplr(np.reshape(buf, (24, 32)))
            max_c  = float(np.max(data))
            mean_c = float(np.mean(data))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(HEATMAP_DIR / f"heatmap_{ts}.png")
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(data, vmin=float(np.min(data)), vmax=max_c, cmap="inferno")
            fig.colorbar(im, ax=ax).set_label("Temperature (C)")
            ax.set_title(f"Thermal {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            return max_c, mean_c, out_path
        except (ValueError, RuntimeError) as e:
            print(f"[THERMAL] attempt {attempt+1}/5: {e}")
            time.sleep(0.05)
    return None, None, None


# ─── GPS THREAD ───────────────────────────────────────────────────────
class GpsReader:
    def __init__(self):
        self.lat = self.lon = self.alt = None
        self._lock = threading.Lock()

    def start(self):
        if _GPS_OK:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    @staticmethod
    def _parse(raw, direction):
        try:
            dot = raw.index(".")
            val = float(raw[:dot-2]) + float(raw[dot-2:]) / 60.0
            return round(-val if direction in ("S","W") else val, 7)
        except Exception:
            return None

    def _run(self):
        while True:
            try:
                line = _GPS_SER.readline().decode("ascii", errors="replace").strip()
                if line.startswith(("$GPGGA","$GNGGA")):
                    p = line.split(",")
                    if len(p) >= 10:
                        lat = self._parse(p[2], p[3])
                        lon = self._parse(p[4], p[5])
                        try: alt = float(p[9]) if p[9] else None
                        except ValueError: alt = None
                        if lat and lon:
                            with self._lock:
                                self.lat, self.lon, self.alt = lat, lon, alt
            except Exception as e:
                print(f"[GPS] Read error: {e}")
                time.sleep(1)

    def get(self):
        with self._lock:
            return self.lat, self.lon, self.alt

    @property
    def has_fix(self):
        with self._lock:
            return self.lat is not None

_gps = GpsReader().start()


# ─── STREAMER ─────────────────────────────────────────────────────────
class HawkEyeStreamer:
    def __init__(self):
        self.frame       = None
        self.frame_lock  = threading.Lock()
        self.infer_queue = queue.Queue(maxsize=1)

        self.persons      = {}
        self.next_id      = 1
        self.persons_lock = threading.Lock()
        self.detected_ids = set()

        self.locked_id     = None
        self.lock_start    = None
        self.lock_progress = 0.0

        self.mode            = "MANUAL"
        self.drop_triggered  = False
        self.drops_completed = 0
        self.drop_log        = []
        self.captures        = []

        self.fps                = 0.0
        self.primary_confidence = 0.0
        self.running            = True

    def start(self):
        threading.Thread(target=self._capture,   daemon=True).start()
        threading.Thread(target=self._inference, daemon=True).start()
        t = time.time()
        while self.frame is None:
            if time.time()-t > 10: print("[CAM] ERROR: No frame after 10s!"); break
            time.sleep(0.1)
        print("[CAM] Stream ready.")
        return self

    # ── Thread 1: Picamera2 capture ───────────────────────────────────
    def _capture(self):
        prev = 0
        fc   = 0
        while self.running:
            try:
                img = _picam.capture_array()        # BGR888 numpy array
                now = time.time()
                self.fps = 1/(now-prev) if prev else 0
                prev = now
                fc  += 1
                with self.frame_lock:
                    self.frame = img.copy()
                if fc % 6 == 0:
                    try: self.infer_queue.put_nowait(img.copy())
                    except queue.Full: pass
            except Exception as e:
                print(f"[CAM] Capture error: {e}")
                time.sleep(0.05)

    # ── Thread 2: YOLO + tracking ─────────────────────────────────────
    def _inference(self):
        time.sleep(3)
        print("[YOLO] Inference thread ready.")
        while self.running:
            try: frame = self.infer_queue.get(timeout=1.0)
            except queue.Empty: continue
            try:
                results = _MODEL(frame, classes=[0], verbose=False, imgsz=320, workers=0)
                raw = []
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        if conf > CONFIDENCE_MIN:
                            raw.append((x1,y1,x2,y2,conf))
                self._update_persons(raw)
                self._update_lock()
                with self.persons_lock:
                    self.primary_confidence = max(
                        (p["conf"] for p in self.persons.values()), default=0.0)
            except Exception as e:
                print(f"[YOLO] Error: {e}"); time.sleep(0.5)

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
        iw=max(0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0,min(ay2,by2)-max(ay1,by1))
        inter=iw*ih
        if inter==0: return 0.0
        ua=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua if ua else 0.0

    def _update_persons(self, raw_boxes):
        new_ids = []
        with self.persons_lock:
            now = time.time(); new_boxes = list(raw_boxes)
            for pid,pdata in self.persons.items():
                pb=(pdata["x1"],pdata["y1"],pdata["x2"],pdata["y2"])
                best_iou=-1; best_box=None; best_idx=-1
                for i,(x1,y1,x2,y2,conf) in enumerate(new_boxes):
                    iou=self._iou(pb,(x1,y1,x2,y2))
                    if iou>best_iou: best_iou=iou; best_box=(x1,y1,x2,y2,conf); best_idx=i
                if best_iou>0.3 and best_box:
                    x1,y1,x2,y2,conf=best_box
                    self.persons[pid].update({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                                              "conf":conf,"last_seen":now})
                    new_boxes.pop(best_idx)
            for (x1,y1,x2,y2,conf) in new_boxes:
                pid=self.next_id; self.next_id+=1
                self.persons[pid]={"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                                   "conf":conf,"label":f"P{pid}","last_seen":now}
                new_ids.append(pid)
            stale=[pid for pid,p in self.persons.items() if now-p["last_seen"]>2.0]
            for pid in stale:
                del self.persons[pid]
                if self.locked_id==pid:
                    self.locked_id=None; self.lock_start=None; self.lock_progress=0.0
        for pid in new_ids:
            if pid not in self.detected_ids:
                self.detected_ids.add(pid)
                threading.Thread(target=self._save_photo,
                                 args=(pid,"detection"), daemon=True).start()

    def _person_in_box(self, p):
        cx=(p["x1"]+p["x2"])//2; cy=(p["y1"]+p["y2"])//2
        return BOX_X1<=cx<=BOX_X2 and BOX_Y1<=cy<=BOX_Y2

    def _update_lock(self):
        with self.persons_lock:
            if self.drop_triggered: return
            target_id = None
            for pid,pdata in self.persons.items():
                if self._person_in_box(pdata): target_id=pid; break
            if target_id is not None:
                if self.locked_id != target_id:
                    self.locked_id=target_id; self.lock_start=time.time(); self.lock_progress=0.0
                else:
                    elapsed = time.time()-self.lock_start
                    self.lock_progress = min(elapsed/LOCK_ON_SECONDS, 1.0)
                    if self.mode=="AUTO" and self.lock_progress>=1.0:
                        self._execute_drop("AUTO", target_id)
            else:
                self.locked_id=None; self.lock_start=None; self.lock_progress=0.0

    def _save_photo(self, pid, event):
        with self.frame_lock:
            if self.frame is None: return None
            snapshot = self.frame.copy()
        lat, lon, alt    = _gps.get()
        max_c, mean_c, _ = _read_thermal()
        ts    = datetime.now()
        tag   = "det" if event=="detection" else "drp"
        fmt   = "%Y%m%d_%H%M%S"
        fname = f"P{pid}_{tag}_{ts.strftime(fmt)}.jpg"
        cv2.imwrite(str(CAPTURES_DIR / fname), snapshot)
        meta = {
            "file": fname, "person_id": pid, "label": f"P{pid}",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "event": event,
            "gps_lat": lat, "gps_lon": lon, "gps_alt": alt,
            "thermo_max_c": max_c, "thermo_mean_c": mean_c,
        }
        (CAPTURES_DIR / fname.replace(".jpg",".json")).write_text(
            json.dumps(meta, indent=2))
        self.captures.insert(0, meta)
        print(f"[{event.upper()}] P{pid} -> {fname} | GPS:{lat},{lon} | Thermo:{max_c}C")
        return meta

    def _execute_drop(self, source, pid=None):
        ts=datetime.now().strftime("%H:%M:%S"); label=f"P{pid}" if pid else "?"
        self.drop_log.insert(0, f"[{ts}] DROP -> {label} ({source})")
        self.drops_completed+=1; self.drop_triggered=True
        print(f"[DROP] #{self.drops_completed} -> {label} ({source})")
        if _DROPPER_OK and _dropper:
            try: _dropper.drop_sequence(); print("[DROP] Servo fired")
            except Exception as e: print(f"[DROP] Servo error: {e}")
        if pid: self._save_photo(pid, "drop")

    def reset_drop(self):
        with self.persons_lock:
            self.drop_triggered=False; self.locked_id=None
            self.lock_start=None; self.lock_progress=0.0
        if _DROPPER_OK and _dropper:
            try: _dropper.hold_position()
            except Exception: pass

    def reset_session(self):
        deleted = 0
        for f in CAPTURES_DIR.glob("*"):
            try: f.unlink(); deleted+=1
            except Exception as e: print(f"[RESET] {f}: {e}")
        with self.persons_lock:
            self.persons={}; self.next_id=1; self.detected_ids=set()
            self.locked_id=None; self.lock_start=None
            self.lock_progress=0.0; self.drop_triggered=False
        self.drops_completed=0; self.drop_log=[]; self.captures=[]
        print(f"[RESET] {deleted} files deleted, IDs reset to P1")

    def get_jpeg(self):
        with self.frame_lock:
            if self.frame is None: return None
            display = self.frame.copy()
        with self.persons_lock:
            persons_snap=dict(self.persons); locked_id=self.locked_id
            lock_progress=self.lock_progress; drop_triggered=self.drop_triggered
        H,W=display.shape[:2]
        is_locked=locked_id is not None and lock_progress>=1.0
        box_color=(0,200,255) if is_locked else (0,220,120)
        corner_len=18
        for (cx,cy,dx,dy) in [(BOX_X1,BOX_Y1,1,1),(BOX_X2,BOX_Y1,-1,1),
                               (BOX_X1,BOX_Y2,1,-1),(BOX_X2,BOX_Y2,-1,-1)]:
            cv2.line(display,(cx,cy),(cx+dx*corner_len,cy),box_color,2)
            cv2.line(display,(cx,cy),(cx,cy+dy*corner_len),box_color,2)
        ov=display.copy()
        cv2.rectangle(ov,(BOX_X1,BOX_Y1),(BOX_X2,BOX_Y2),box_color,1)
        cv2.addWeighted(ov,0.15,display,0.85,0,display)
        if lock_progress>0 and not drop_triggered:
            bar_w=int(BOX_W*lock_progress)
            bar_color=(0,200,255) if lock_progress>=1.0 else (0,255,200)
            cv2.rectangle(display,(BOX_X1,BOX_Y2+4),(BOX_X2,BOX_Y2+10),(30,30,30),-1)
            cv2.rectangle(display,(BOX_X1,BOX_Y2+4),(BOX_X1+bar_w,BOX_Y2+10),bar_color,-1)
        for pid,p in persons_snap.items():
            x1,y1,x2,y2=p["x1"],p["y1"],p["x2"],p["y2"]
            locked_this=(pid==locked_id)
            color=(0,200,255) if locked_this else (0,220,80)
            cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
            label=p.get("label",f"P{pid}")
            if locked_this and lock_progress>0:
                label=f"{label}  {int(lock_progress*100)}%"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(display,(x1,y1-th-6),(x1+tw+6,y1),color,-1)
            cv2.putText(display,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
            if locked_this:
                lx=(x1+x2)//2; ly=(y1+y2)//2; arm=14
                cv2.line(display,(lx-arm,ly),(lx+arm,ly),(0,200,255),1)
                cv2.line(display,(lx,ly-arm),(lx,ly+arm),(0,200,255),1)
                cv2.line(display,(lx,ly),(W//2,H//2),(0,200,255),1)
        cv2.rectangle(display,(0,0),(W,32),(15,15,15),-1)
        mode_color=(0,180,255) if self.mode=="AUTO" else (80,220,80)
        cv2.putText(display,f"MODE:{self.mode}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,mode_color,1)
        cv2.putText(display,f"FPS:{self.fps:.1f}",(150,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
        cv2.putText(display,f"PERSONS:{len(persons_snap)}",(270,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
        cv2.putText(display,f"DROPS:{self.drops_completed}",(430,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
        lat,lon,_=_gps.get()
        if lat and lon:
            cv2.putText(display,f"GPS {lat:.5f},{lon:.5f}",(8,H-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,220,120),1)
        if drop_triggered:
            cv2.putText(display,"PAYLOAD RELEASED",(W//2-140,H//2),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,80,255),3)
        elif is_locked:
            cv2.putText(display,"LOCKED ON",(W//2-80,H-40),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,200,255),2)
        _,buf=cv2.imencode(".jpg",display,[cv2.IMWRITE_JPEG_QUALITY,70])
        return buf.tobytes()

    def stop(self):
        self.running=False
        _picam.stop()

    def get_status(self):
        with self.persons_lock:
            n=len(self.persons); locked_id=self.locked_id; lp=self.lock_progress
            is_locked=locked_id is not None and lp>=1.0
        lat,lon,alt=_gps.get()
        return {
            "mode": self.mode, "fps": round(self.fps,1),
            "persons_count": n, "confidence": round(self.primary_confidence,2),
            "is_locked": is_locked, "locked_id": locked_id,
            "lock_progress": round(lp,2),
            "drop_triggered": self.drop_triggered,
            "drops_completed": self.drops_completed,
            "lat": lat, "lon": lon, "alt": alt,
            "gps_fix": _gps.has_fix,
            "thermal_ok": _THERMAL_OK,
            "dropper_ok": _DROPPER_OK,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "log": self.drop_log[:20],
            "captures": self.captures[:20],
        }

# ─── INIT ─────────────────────────────────────────────────────────────
video_stream = HawkEyeStreamer().start()

# ─── HTML ─────────────────────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HawkEye — Rescue Drone Module</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body {
            background:#0a0e17; color:#e0e0e0;
            font-family:'Segoe UI',system-ui,sans-serif;
            height:100vh; overflow:hidden;
        }

        /* ── HEADER ── */
        .header {
            background:linear-gradient(135deg,#0d1520,#142030);
            padding:8px 20px;
            display:flex; justify-content:space-between; align-items:center;
            border-bottom:1px solid #1a3050;
        }
        .header-left { display:flex; align-items:center; gap:10px; }
        .logo-ring {
            width:30px; height:30px; border-radius:50%;
            border:2px solid #e53935;
            display:flex; align-items:center; justify-content:center;
            color:#e53935; font-size:14px;
        }
        .header h1 { font-size:16px; font-weight:600; color:#5ba3e6; letter-spacing:2px; }
        .header-sub { font-size:10px; color:#2e5a75; letter-spacing:2px; margin-top:1px; }
        .header-right { display:flex; align-items:center; gap:10px; }
        .gallery-btn {
            padding:4px 14px; border-radius:20px; font-size:11px; font-weight:600;
            letter-spacing:1px; text-decoration:none; cursor:pointer;
            background:#0a2535; color:#38bdf8; border:1px solid #1a3a55;
            transition:background 0.15s;
        }
        .gallery-btn:hover { background:#112535; }
        .status-pill {
            padding:3px 14px; border-radius:20px;
            font-size:11px; font-weight:600; letter-spacing:1px;
            background:#0a3d1a; color:#4ade80; border:1px solid #166534;
        }

        /* ── LAYOUT ── */
        .main {
            display:grid;
            grid-template-columns:1fr 300px;
            height:calc(100vh - 48px);
        }

        /* ── VIDEO ── */
        .video-container {
            background:#000; display:flex;
            align-items:center; justify-content:center; overflow:hidden;
        }
        .video-container img { width:100%; height:100%; object-fit:contain; }

        /* ── PANEL ── */
        .panel {
            background:#0d1520; border-left:1px solid #1a3050;
            padding:12px; display:flex; flex-direction:column;
            gap:10px; overflow-y:auto;
        }
        .panel::-webkit-scrollbar { width:3px; }
        .panel::-webkit-scrollbar-thumb { background:#1a3050; border-radius:2px; }

        .card {
            background:#111b2a; border:1px solid #1a3050;
            border-radius:8px; padding:12px; flex-shrink:0;
        }
        .card-title {
            font-size:10px; text-transform:uppercase;
            letter-spacing:1.5px; color:#5ba3e6;
            margin-bottom:9px; font-weight:600;
            display:flex; align-items:center; gap:6px;
        }
        .card-title .badge {
            font-size:9px; padding:1px 5px; border-radius:3px;
            background:#1a2535; color:#4a5568;
        }

        /* ── STATS ── */
        .stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
        .stat-box {
            background:#0a1018; border-radius:6px;
            padding:8px; text-align:center;
        }
        .stat-value { font-size:20px; font-weight:700; color:#fff; }
        .stat-label { font-size:9px; color:#4a5568; text-transform:uppercase; letter-spacing:1px; margin-top:1px; }

        /* ── MODE ── */
        .mode-toggle {
            display:flex; background:#0a1018;
            border-radius:6px; overflow:hidden; border:1px solid #1a3050;
        }
        .mode-btn {
            flex:1; padding:10px; text-align:center;
            font-size:12px; font-weight:600; letter-spacing:1px;
            cursor:pointer; border:none; color:#4a5568;
            background:transparent; transition:all 0.2s;
        }
        .mode-btn.active-manual { background:#0a3d1a; color:#4ade80; }
        .mode-btn.active-auto   { background:#3d1a0a; color:#f87171; }

        /* ── LOCK BAR ── */
        .lock-bar-container { background:#0a1018; border-radius:6px; padding:8px; }
        .lock-bar-label {
            display:flex; justify-content:space-between;
            font-size:10px; color:#4a5568; margin-bottom:5px;
        }
        .lock-bar-track { height:7px; background:#1a2535; border-radius:4px; overflow:hidden; }
        .lock-bar-fill {
            height:100%; border-radius:4px;
            background:linear-gradient(90deg,#fbbf24,#f59e0b);
            transition:width 0.15s; width:0%;
        }
        .lock-bar-fill.locked { background:linear-gradient(90deg,#f97316,#ef4444); }

        /* ── DROP BUTTON ── */
        .drop-btn {
            width:100%; padding:14px; border-radius:8px;
            font-size:13px; font-weight:700; letter-spacing:2px;
            cursor:pointer; transition:all 0.2s; border:2px solid transparent;
        }
        .drop-ready   { background:linear-gradient(135deg,#dc2626,#b91c1c); color:#fff; border-color:#ef4444; }
        .drop-ready:hover { background:linear-gradient(135deg,#ef4444,#dc2626); transform:scale(1.02); }
        .drop-waiting { background:#1a2535; color:#4a5568; border-color:#2d3748; cursor:not-allowed; }
        .drop-done    { background:#0a3d1a; color:#4ade80; border-color:#166534; cursor:not-allowed; }
        .reset-btn {
            width:100%; padding:8px; border-radius:6px; margin-top:6px;
            font-size:11px; font-weight:600; letter-spacing:1px; cursor:pointer;
            background:#1a2535; color:#5ba3e6; border:1px solid #1a3050; transition:all 0.2s;
        }
        .reset-btn:hover { background:#1f3045; }
        .new-mission-btn {
            width:100%; padding:10px; border-radius:6px;
            font-size:11px; font-weight:700; letter-spacing:1px; cursor:pointer;
            background:#1a0800; color:#f97316; border:1px solid #3a1800; transition:all 0.2s;
        }
        .new-mission-btn:hover  { background:#2d1200; border-color:#f97316; }
        .new-mission-btn:active { background:#f97316; color:#000; }

        /* ── GPS ── */
        .gps-text { font-family:'Consolas',monospace; font-size:12px; color:#5ba3e6; line-height:1.9; }
        .gps-pending { color:#2e5a75; font-style:italic; }

        /* ── LOG ── */
        .log-box {
            max-height:90px; overflow-y:auto;
            font-family:'Consolas',monospace; font-size:10px; line-height:1.7; color:#4a5568;
        }
        .log-box::-webkit-scrollbar { width:2px; }
        .log-box::-webkit-scrollbar-thumb { background:#1a3050; }
        .log-drop     { color:#f87171; }
        .log-lock     { color:#fbbf24; }
        .log-detect   { color:#4ade80; }
        .log-detection { color:#38bdf8; }

        /* ── SHARED CAP/DETECT LIST STYLES ── */
        .event-list { max-height:110px; overflow-y:auto; font-size:10px; }
        .event-list::-webkit-scrollbar { width:2px; }
        .event-list::-webkit-scrollbar-thumb { background:#1a3050; }
        .event-row {
            padding:4px 0; border-bottom:1px solid #0d1a24;
            display:grid; gap:3px;
        }
        .event-row:last-child { border-bottom:none; }
        .event-header { display:flex; justify-content:space-between; align-items:center; }
        .event-label  { font-weight:600; }
        .event-label.drop-label      { color:#f87171; }
        .event-label.detection-label { color:#38bdf8; }
        .event-time   { color:#2e5a75; font-size:9px; font-family:'Consolas',monospace; }
        .event-meta   { display:flex; gap:8px; }
        .event-tag {
            font-size:8px; padding:1px 5px; border-radius:3px; font-family:'Consolas',monospace;
            letter-spacing:0.5px;
        }
        .tag-gps     { background:#0d1f30; color:#2e5a75; }
        .tag-thermo  { background:#1a1010; color:#4a2020; }
        .tag-gps.has-data    { background:#0a2535; color:#38bdf8; }
        .tag-thermo.has-data { background:#2d1000; color:#f97316; }
        .event-file { color:#1a3050; font-size:9px; font-family:'Consolas',monospace;
                      overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

        /* ── TAB SWITCHER for detections/captures ── */
        .tab-row { display:flex; gap:0; margin-bottom:9px; border-radius:5px; overflow:hidden; border:1px solid #1a3050; }
        .tab-btn {
            flex:1; padding:5px 0; text-align:center; font-size:10px; font-weight:600;
            letter-spacing:1px; cursor:pointer; border:none; background:#0a1018; color:#2e5a75;
            transition:all 0.15s; text-transform:uppercase;
        }
        .tab-btn.active { background:#1a2535; color:#5ba3e6; }
        .tab-pane { display:none; }
        .tab-pane.active { display:block; }
    </style>
</head>
<body>
<div class="header">
    <div class="header-left">
        <div class="logo-ring">✦</div>
        <div>
            <h1>HAWKEYE</h1>
            <div class="header-sub">RESCUE DRONE MODULE v1.0</div>
        </div>
    </div>
    <div class="header-right">
        <a href="/gallery" target="_blank" class="gallery-btn">📷 GALLERY</a>
        <span class="status-pill">● CONNECTED</span>
    </div>
</div>

<div class="main">
    <!-- Video -->
    <div class="video-container">
        <img id="videoFeed" src="/video_feed" alt="Live Feed">
    </div>

    <!-- Panel -->
    <div class="panel">

        <!-- Telemetry -->
        <div class="card">
            <div class="card-title">Live Telemetry</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="fpsVal">0</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="targetsVal">0</div>
                    <div class="stat-label">Persons</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="confVal">0%</div>
                    <div class="stat-label">Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="dropsVal">0</div>
                    <div class="stat-label">Drops</div>
                </div>
            </div>
        </div>

        <!-- Mode -->
        <div class="card">
            <div class="card-title">Drop Mode</div>
            <div class="mode-toggle">
                <button class="mode-btn active-manual" id="btnManual" onclick="setMode('MANUAL')">MANUAL</button>
                <button class="mode-btn" id="btnAuto" onclick="setMode('AUTO')">AUTO</button>
            </div>
        </div>

        <!-- Lock -->
        <div class="card">
            <div class="card-title">Target Lock</div>
            <div class="lock-bar-container">
                <div class="lock-bar-label">
                    <span id="lockStatus">Scanning...</span>
                    <span id="lockPct">0%</span>
                </div>
                <div class="lock-bar-track">
                    <div class="lock-bar-fill" id="lockBar"></div>
                </div>
            </div>
        </div>

        <!-- Drop -->
        <div class="card">
            <div class="card-title">Payload Release</div>
            <button class="drop-btn drop-waiting" id="dropBtn" onclick="triggerDrop()">
                WAITING FOR LOCK
            </button>
            <button class="reset-btn" id="resetBtn" onclick="resetDrop()" style="display:none">
                ↻ RESET FOR NEXT TARGET
            </button>
        </div>

        <!-- GPS + Sensor Status -->
        <div class="card">
            <div class="card-title">
                GPS Position
                <span class="badge" id="gpsBadge">NO FIX</span>
            </div>
            <div class="gps-text">
                LAT: <span id="latVal" class="gps-pending">--</span><br>
                LON: <span id="lonVal" class="gps-pending">--</span><br>
                ALT: <span id="altVal" class="gps-pending">--</span><br>
                TIME: <span id="timeVal">--</span>
            </div>
            <div style="display:flex;gap:5px;margin-top:8px;flex-wrap:wrap">
                <span class="event-tag tag-gps" id="thermalStatus">🌡 Thermal: --</span>
                <span class="event-tag tag-gps" id="dropperStatus">⚙ Servos: --</span>
            </div>
        </div>

        <!-- Log -->
        <div class="card">
            <div class="card-title">Mission Log</div>
            <div class="log-box" id="missionLog">
                <div style="color:#2e3a4a">Awaiting events...</div>
            </div>
        </div>

        <!-- New Mission button -->
        <div class="card" style="border-color:#2d1500">
            <div class="card-title" style="color:#f97316">⚠ Mission Control</div>
            <button class="new-mission-btn" id="newMissionBtn" onclick="confirmReset()">
                🔄 NEW MISSION — RESET P1
            </button>
            <div style="font-size:9px;color:#3a2010;margin-top:5px;line-height:1.6;font-family:Consolas,monospace">
                Resets person IDs to P1<br>Deletes all capture photos &amp; logs
            </div>
        </div>

        <!-- Captures (unified list) -->
        <div class="card">
            <div class="card-title">
                📸 Captures
                <span class="badge" id="capCount">0</span>
                <a href="/gallery" target="_blank" style="margin-left:auto;font-size:9px;color:#38bdf8;text-decoration:none;letter-spacing:1px">VIEW ALL →</a>
            </div>
            <div class="event-list" id="captureList">
                <div style="color:#2e3a4a;font-size:10px">No captures yet.</div>
            </div>
        </div>

    </div>
</div>

<script>
    let wasLocked=false, wasDropped=false;
    let lastCapLen=0, lastLogLen=0;

    // Build a capture row (works for both detection and drop events)
    function buildEventRow(c) {
        const row = document.createElement('div');
        row.className = 'event-row';
        row.dataset.file = c.file;

        const isDetect  = c.event === 'detection';
        const hasGps    = c.gps_lat != null && c.gps_lon != null;
        const hasThermo = c.thermo_max_c != null;
        const gpsText   = hasGps    ? c.gps_lat.toFixed(5)+', '+c.gps_lon.toFixed(5) : 'GPS pending';
        const thText    = hasThermo ? c.thermo_max_c.toFixed(1)+'°C' : 'Thermal pending';

        row.innerHTML =
            '<div class="event-header">'
          +   '<span class="event-label '+(isDetect?'detection-label':'drop-label')+'">'+c.label+'</span>'
          +   '<span class="event-badge badge-'+(isDetect?'detection':'drop')+'">'+(isDetect?'SEEN':'DROP')+'</span>'
          +   '<span class="event-time">'+c.timestamp.split(' ')[1]+'</span>'
          + '</div>'
          + '<div class="event-meta">'
          +   '<span class="event-tag tag-gps'+(hasGps?' has-data':'')+'">📍 '+gpsText+'</span>'
          +   '<span class="event-tag tag-thermo'+(hasThermo?' has-data':'')+'">🌡 '+thText+'</span>'
          + '</div>'
          + '<div class="event-file">'+c.file+'</div>';
        return row;
    }

    async function pollStatus() {
        try {
            const d = await (await fetch('/status')).json();

            document.getElementById('fpsVal').textContent     = d.fps;
            document.getElementById('targetsVal').textContent = d.persons_count;
            document.getElementById('confVal').textContent    = Math.round(d.confidence*100)+'%';
            document.getElementById('dropsVal').textContent   = d.drops_completed;
            document.getElementById('timeVal').textContent    = d.timestamp;

            // GPS — live values from NEO-6M
            if (d.gps_fix && d.lat != null) {
                document.getElementById('latVal').textContent = d.lat.toFixed(6);
                document.getElementById('lonVal').textContent = d.lon.toFixed(6);
                document.getElementById('altVal').textContent = d.alt != null ? d.alt.toFixed(1)+' m' : '--';
                document.getElementById('latVal').classList.remove('gps-pending');
                document.getElementById('lonVal').classList.remove('gps-pending');
                document.getElementById('altVal').classList.remove('gps-pending');
                document.getElementById('gpsBadge').textContent = 'FIX ✓';
                document.getElementById('gpsBadge').style.color = '#4ade80';
            } else {
                document.getElementById('gpsBadge').textContent = 'NO FIX';
                document.getElementById('gpsBadge').style.color = '#f87171';
            }
            // Sensor status pills
            const thEl = document.getElementById('thermalStatus');
            thEl.textContent = '🌡 Thermal: ' + (d.thermal_ok ? 'OK' : 'OFFLINE');
            thEl.className   = 'event-tag ' + (d.thermal_ok ? 'tag-thermo has-data' : 'tag-thermo');
            const drEl = document.getElementById('dropperStatus');
            drEl.textContent = '⚙ Servos: ' + (d.dropper_ok ? 'OK' : 'OFFLINE');
            drEl.className   = 'event-tag ' + (d.dropper_ok ? 'tag-gps has-data' : 'tag-gps');

            // Lock bar
            const bar=document.getElementById('lockBar');
            const ls =document.getElementById('lockStatus');
            const lp =document.getElementById('lockPct');
            if (d.is_locked) {
                bar.style.width='100%'; bar.classList.add('locked');
                ls.textContent='LOCKED ON — P'+(d.locked_id||'?'); lp.textContent='100%';
                if (!wasLocked) { addLog('Target locked: P'+d.locked_id,'lock'); wasLocked=true; }
            } else if (d.lock_progress>0) {
                bar.style.width=(d.lock_progress*100)+'%'; bar.classList.remove('locked');
                ls.textContent='Locking P'+(d.locked_id||'')+'...';
                lp.textContent=Math.round(d.lock_progress*100)+'%'; wasLocked=false;
            } else {
                bar.style.width='0%'; bar.classList.remove('locked');
                ls.textContent=d.persons_count>0?'Person detected — center to lock':'Scanning...';
                lp.textContent='0%'; wasLocked=false;
            }

            // Drop button
            const btn =document.getElementById('dropBtn');
            const rbtn=document.getElementById('resetBtn');
            if (d.drop_triggered) {
                btn.className='drop-btn drop-done'; btn.textContent='✓ PAYLOAD RELEASED'; btn.disabled=true;
                rbtn.style.display='block';
                if (!wasDropped) { addLog('Drop #'+d.drops_completed+' complete','drop'); wasDropped=true; }
            } else if (d.is_locked) {
                btn.className='drop-btn drop-ready'; btn.textContent='🎯 RELEASE PAYLOAD'; btn.disabled=false;
                rbtn.style.display='none'; wasDropped=false;
            } else {
                btn.className='drop-btn drop-waiting'; btn.textContent='WAITING FOR LOCK'; btn.disabled=true;
                rbtn.style.display='none'; wasDropped=false;
            }

            // Mode buttons
            document.getElementById('btnManual').className='mode-btn'+(d.mode==='MANUAL'?' active-manual':'');
            document.getElementById('btnAuto').className  ='mode-btn'+(d.mode==='AUTO'  ?' active-auto'  :'');

            // Mission log
            if (d.log.length > lastLogLen) {
                d.log.slice(0, d.log.length-lastLogLen).forEach(e=>addLog(e,'drop'));
                lastLogLen = d.log.length;
            }

            // Unified captures list (detection + drop)
            if (d.captures.length > lastCapLen) {
                const cl = document.getElementById('captureList');
                if (cl.querySelector('div[style]')) cl.innerHTML='';
                d.captures.slice(lastCapLen).forEach(c => {
                    if (document.querySelector('[data-file="'+c.file+'"]')) return;
                    cl.prepend(buildEventRow(c));
                });
                lastCapLen = d.captures.length;
                document.getElementById('capCount').textContent = d.captures.length;
                const newest = d.captures[0];
                addLog((newest.event==='detection'?'📡 Detected: ':'📦 Drop photo: ')+newest.label,
                       newest.event==='detection'?'detection':'drop');
            }

        } catch(e) { console.error(e); }
    }

    function addLog(msg, type) {
        const log=document.getElementById('missionLog');
        if (log.querySelector('[style]')) log.innerHTML='';
        const el=document.createElement('div');
        el.className = type==='drop'  ? 'log-drop'
                     : type==='lock'  ? 'log-lock'
                     : type==='detection' ? 'log-detection'
                     : 'log-detect';
        el.textContent='['+new Date().toLocaleTimeString()+'] '+msg;
        log.prepend(el);
        while(log.children.length>30) log.removeChild(log.lastChild);
    }

    async function setMode(m) {
        await fetch('/set_mode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
        addLog('Mode → '+m,'detect');
    }

    async function triggerDrop() {
        await fetch('/drop',{method:'POST'});
    }

    async function resetDrop() {
        await fetch('/reset_drop',{method:'POST'});
        addLog('System reset — ready for next target','detect');
    }

    async function confirmReset() {
        // Simple confirm — prevents accidental taps
        const btn = document.getElementById('newMissionBtn');
        if (btn.dataset.confirm !== 'yes') {
            btn.textContent = '⚠ TAP AGAIN TO CONFIRM';
            btn.style.borderColor = '#ef4444';
            btn.style.color = '#ef4444';
            btn.dataset.confirm = 'yes';
            setTimeout(() => {
                btn.textContent = '🔄 NEW MISSION — RESET P1';
                btn.style.borderColor = '';
                btn.style.color = '';
                btn.dataset.confirm = '';
            }, 3000);
            return;
        }
        // Confirmed — do the reset
        btn.textContent = 'RESETTING...';
        btn.disabled = true;
        try {
            await fetch('/reset_session', {method:'POST'});
            lastCapLen  = 0;
            lastLogLen  = 0;
            wasLocked   = false;
            wasDropped  = false;
            // Clear captures list in UI
            const cl = document.getElementById('captureList');
            cl.innerHTML = '<div style="color:#2e3a4a;font-size:10px">No captures yet.</div>';
            document.getElementById('capCount').textContent = '0';
            // Clear log
            document.getElementById('missionLog').innerHTML = '<div style="color:#2e3a4a">Awaiting events...</div>';
            addLog('🔄 New mission started — IDs reset to P1', 'detect');
        } catch(e) {
            addLog('Reset failed: '+e, 'drop');
        }
        btn.textContent = '🔄 NEW MISSION — RESET P1';
        btn.style.borderColor = '';
        btn.style.color = '';
        btn.dataset.confirm = '';
        btn.disabled = false;
    }

    setInterval(pollStatus, 250);
    pollStatus();
</script>
</body>
</html>
"""

# ─── ROUTES ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

def gen_frames():
    while True:
        frame = video_stream.get_jpeg()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(video_stream.get_status())

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    if data and data.get('mode') in ('AUTO','MANUAL'):
        video_stream.mode = data['mode']
        print(f"[MODE] → {video_stream.mode}")
    return jsonify({"mode": video_stream.mode})

@app.route('/drop', methods=['POST'])
def drop():
    with video_stream.persons_lock:
        pid = video_stream.locked_id
        ok  = video_stream.drop_triggered is False and pid is not None
    if ok and video_stream.mode == 'MANUAL':
        video_stream._execute_drop("MANUAL", pid)
    return jsonify({"ok": ok})

@app.route('/reset_drop', methods=['POST'])
def reset_drop():
    video_stream.reset_drop()
    return jsonify({"ok": True})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    video_stream.reset_session()
    return jsonify({"ok": True, "message": "Session reset — IDs start from P1"})

@app.route('/captures/<path:filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURES_DIR, filename)

@app.route('/api/gallery')
def api_gallery():
    """Scan both folders and return all images with sidecar JSON metadata."""
    items = []

    for img_path in sorted(CAPTURES_DIR.glob("*.jpg"), reverse=True):
        json_path = img_path.with_suffix('.json')
        meta = {}
        if json_path.exists():
            try:
                meta = json.loads(json_path.read_text())
            except Exception:
                pass
        items.append({
            "url":       f"/captures/{img_path.name}",
            "filename":  img_path.name,
            "event":     meta.get("event", "unknown"),
            "label":     meta.get("label", img_path.stem),
            "person_id": meta.get("person_id"),
            "timestamp": meta.get("timestamp", ""),
            "gps_lat":   meta.get("gps_lat"),
            "gps_lon":   meta.get("gps_lon"),
            "gps_alt":   meta.get("gps_alt"),
            "thermo_max_c":  meta.get("thermo_max_c"),
            "thermo_mean_c": meta.get("thermo_mean_c"),
        })

    # Sort newest first by timestamp string (ISO-like format sorts correctly)
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(items)

GALLERY_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HawkEye — Gallery</title>
    <style>
        :root {
            --bg:      #0a0e17;
            --bg2:     #0d1520;
            --bg3:     #111b2a;
            --border:  #1a3050;
            --accent:  #5ba3e6;
            --green:   #4ade80;
            --red:     #f87171;
            --blue:    #38bdf8;
            --orange:  #f97316;
            --dim:     #2e5a75;
            --dimmer:  #1a3050;
            --text:    #e0e0e0;
            --mono:    'Consolas', 'Courier New', monospace;
        }

        * { margin:0; padding:0; box-sizing:border-box; }
        body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif; min-height:100vh; }

        /* ── HEADER ── */
        .header {
            background:linear-gradient(135deg,#0d1520,#142030);
            padding:8px 20px; display:flex; justify-content:space-between; align-items:center;
            border-bottom:1px solid var(--border); position:sticky; top:0; z-index:100;
        }
        .header-left { display:flex; align-items:center; gap:12px; }
        .logo-ring {
            width:30px; height:30px; border-radius:50%; border:2px solid #e53935;
            display:flex; align-items:center; justify-content:center; color:#e53935; font-size:14px;
        }
        .title-block h1 { font-size:16px; font-weight:600; color:var(--accent); letter-spacing:2px; }
        .title-block .sub { font-size:10px; color:var(--dim); letter-spacing:2px; margin-top:1px; }
        .header-right { display:flex; align-items:center; gap:10px; }
        .back-btn {
            padding:5px 14px; border-radius:20px; font-size:11px; font-weight:600;
            border:1px solid var(--border); background:var(--bg2); color:var(--accent);
            text-decoration:none; letter-spacing:1px; transition:all 0.2s;
        }
        .back-btn:hover { background:var(--border); }
        .count-pill {
            padding:3px 12px; border-radius:20px; font-size:11px;
            background:#0a2535; color:var(--blue); border:1px solid #1a3a55;
            font-family:var(--mono); letter-spacing:1px;
        }

        /* ── TOOLBAR ── */
        .toolbar {
            padding:12px 20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap;
            border-bottom:1px solid var(--border); background:var(--bg2);
        }
        .filter-btn {
            padding:5px 14px; border-radius:4px; font-size:11px; font-weight:600;
            letter-spacing:1px; cursor:pointer; border:1px solid var(--border);
            background:var(--bg); color:var(--dim); transition:all 0.15s;
        }
        .filter-btn.active     { background:#0a2535; color:var(--blue);  border-color:#1a3a55; }
        .filter-btn.active-all { background:#1a2535; color:var(--text);  border-color:var(--border); }
        .filter-btn.active-det { background:#092030; color:var(--blue);  border-color:#1a3a55; }
        .filter-btn.active-drop{ background:#200a0a; color:var(--red);   border-color:#3a1515; }
        .sort-select {
            padding:5px 10px; border-radius:4px; font-size:11px; font-family:var(--mono);
            background:var(--bg); color:var(--dim); border:1px solid var(--border); cursor:pointer;
        }
        .refresh-btn {
            padding:5px 12px; border-radius:4px; font-size:11px; font-weight:600;
            letter-spacing:1px; cursor:pointer; border:1px solid var(--border);
            background:var(--bg); color:var(--accent); transition:all 0.15s; margin-left:auto;
        }
        .refresh-btn:hover { background:var(--border); }
        .toolbar-label { font-size:10px; color:var(--dimmer); letter-spacing:1px; text-transform:uppercase; }

        /* ── GALLERY GRID ── */
        .gallery {
            padding:16px 20px;
            display:grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap:14px;
        }

        /* ── CARD ── */
        .img-card {
            background:var(--bg3); border:1px solid var(--border);
            border-radius:8px; overflow:hidden; cursor:pointer;
            transition:transform 0.15s, border-color 0.15s;
            display:flex; flex-direction:column;
        }
        .img-card:hover { transform:translateY(-2px); border-color:var(--accent); }
        .img-card.type-detection { border-left:3px solid var(--blue); }
        .img-card.type-drop      { border-left:3px solid var(--red);  }

        .img-thumb {
            width:100%; aspect-ratio:4/3; object-fit:cover; display:block;
            background:#000; transition:opacity 0.2s;
        }
        .img-thumb:hover { opacity:0.85; }

        .img-info { padding:10px; display:flex; flex-direction:column; gap:5px; }

        .img-header { display:flex; justify-content:space-between; align-items:center; }
        .img-label  { font-size:13px; font-weight:700; }
        .img-label.detection { color:var(--blue); }
        .img-label.drop      { color:var(--red);  }
        .event-badge {
            font-size:9px; padding:2px 7px; border-radius:3px; font-weight:600;
            letter-spacing:1px; text-transform:uppercase;
        }
        .badge-detection { background:#092030; color:var(--blue); border:1px solid #1a3a55; }
        .badge-drop      { background:#200a0a; color:var(--red);  border:1px solid #3a1515; }

        .img-time { font-family:var(--mono); font-size:10px; color:var(--dim); }

        .img-tags { display:flex; gap:5px; flex-wrap:wrap; }
        .tag {
            font-size:9px; padding:2px 7px; border-radius:3px;
            font-family:var(--mono); letter-spacing:0.5px;
        }
        .tag-gps-empty    { background:#0d1a24; color:var(--dimmer); }
        .tag-gps-live     { background:#0a2535; color:var(--blue);   border:1px solid #1a3a55; }
        .tag-thermo-empty { background:#1a1010; color:#2a1515; }
        .tag-thermo-live  { background:#2d1000; color:var(--orange); border:1px solid #4a2000; }

        .img-file { font-family:var(--mono); font-size:9px; color:var(--dimmer);
                    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

        /* ── EMPTY STATE ── */
        .empty-state {
            grid-column:1/-1; text-align:center; padding:60px 20px; color:var(--dimmer);
        }
        .empty-state .icon { font-size:40px; margin-bottom:12px; opacity:0.4; }
        .empty-state .msg  { font-size:14px; }
        .empty-state .sub  { font-size:11px; margin-top:6px; color:#1a3050; }

        /* ── LIGHTBOX ── */
        .lightbox {
            display:none; position:fixed; inset:0; z-index:200;
            background:rgba(5,8,14,0.95); align-items:center; justify-content:center;
            flex-direction:column; gap:0;
        }
        .lightbox.open { display:flex; }

        .lb-top {
            width:100%; max-width:900px; display:flex; justify-content:space-between;
            align-items:center; padding:12px 16px; flex-shrink:0;
        }
        .lb-label { font-size:16px; font-weight:700; color:var(--accent); }
        .lb-close {
            width:32px; height:32px; border-radius:50%; border:1px solid var(--border);
            background:var(--bg2); color:var(--text); cursor:pointer; font-size:16px;
            display:flex; align-items:center; justify-content:center; transition:all 0.15s;
        }
        .lb-close:hover { background:var(--border); }

        .lb-body {
            width:100%; max-width:900px; display:grid;
            grid-template-columns:1fr 280px; gap:0; flex:1; overflow:hidden;
            border:1px solid var(--border); border-radius:8px; background:var(--bg3);
        }

        .lb-img-wrap {
            background:#000; display:flex; align-items:center; justify-content:center;
            overflow:hidden; min-height:300px;
        }
        .lb-img { max-width:100%; max-height:70vh; object-fit:contain; display:block; }

        .lb-meta {
            padding:16px; border-left:1px solid var(--border);
            overflow-y:auto; display:flex; flex-direction:column; gap:14px;
        }
        .lb-meta::-webkit-scrollbar { width:2px; }
        .lb-meta::-webkit-scrollbar-thumb { background:var(--border); }

        .meta-section {}
        .meta-title {
            font-size:9px; text-transform:uppercase; letter-spacing:2px;
            color:var(--dim); margin-bottom:7px; font-weight:600;
        }
        .meta-row {
            display:flex; justify-content:space-between; align-items:flex-start;
            padding:4px 0; border-bottom:1px solid #0d1a24; gap:8px;
        }
        .meta-row:last-child { border-bottom:none; }
        .meta-key { font-size:10px; color:var(--dim); flex-shrink:0; }
        .meta-val { font-size:10px; font-family:var(--mono); color:var(--accent);
                    text-align:right; word-break:break-all; }
        .meta-val.pending { color:var(--dimmer); font-style:italic; }
        .meta-val.live    { color:var(--blue); }
        .meta-val.thermo  { color:var(--orange); }

        .lb-nav {
            display:flex; gap:10px; margin-top:12px; flex-shrink:0;
            max-width:900px; width:100%; padding:0 16px;
            justify-content:space-between;
        }
        .nav-btn {
            padding:7px 20px; border-radius:4px; font-size:11px; font-weight:600;
            letter-spacing:1px; cursor:pointer; border:1px solid var(--border);
            background:var(--bg2); color:var(--text); transition:all 0.15s;
        }
        .nav-btn:hover  { background:var(--border); }
        .nav-btn:disabled { opacity:0.3; cursor:not-allowed; }
        .nav-pos { font-size:11px; color:var(--dim); font-family:var(--mono); align-self:center; }

        /* ── LOADING ── */
        .loading { text-align:center; padding:60px; color:var(--dim); font-size:13px; }
        @keyframes spin { to { transform:rotate(360deg); } }
        .spinner { display:inline-block; width:20px; height:20px; border:2px solid var(--border);
                   border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite;
                   margin-bottom:10px; }

        @media (max-width:600px) {
            .lb-body { grid-template-columns:1fr; }
            .lb-meta { border-left:none; border-top:1px solid var(--border); max-height:200px; }
            .gallery { grid-template-columns:1fr 1fr; gap:8px; padding:10px; }
        }
    </style>
</head>
<body>

<!-- Header -->
<div class="header">
    <div class="header-left">
        <div class="logo-ring">✦</div>
        <div class="title-block">
            <h1>HAWKEYE — GALLERY</h1>
            <div class="sub">DETECTION &amp; DROP RECORD</div>
        </div>
    </div>
    <div class="header-right">
        <span class="count-pill" id="totalCount">0 photos</span>
        <a href="/" class="back-btn">← LIVE VIEW</a>
    </div>
</div>

<!-- Toolbar -->
<div class="toolbar">
    <span class="toolbar-label">Filter:</span>
    <button class="filter-btn active-all active" id="f-all"  onclick="setFilter('all')">All</button>
    <button class="filter-btn" id="f-detection" onclick="setFilter('detection')">📡 Detections</button>
    <button class="filter-btn" id="f-drop"      onclick="setFilter('drop')">📦 Drops</button>
    <span class="toolbar-label" style="margin-left:8px">Sort:</span>
    <select class="sort-select" id="sortSelect" onchange="applySort()">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="person">Person ID</option>
    </select>
    <button class="refresh-btn" onclick="loadGallery()">↻ Refresh</button>
</div>

<!-- Gallery -->
<div class="gallery" id="gallery">
    <div class="loading"><div class="spinner"></div><br>Loading captures...</div>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightboxOutside(event)">
    <div class="lb-top">
        <span class="lb-label" id="lbLabel">—</span>
        <button class="lb-close" onclick="closeLightbox()">✕</button>
    </div>
    <div class="lb-body">
        <div class="lb-img-wrap">
            <img class="lb-img" id="lbImg" src="" alt="">
        </div>
        <div class="lb-meta" id="lbMeta"></div>
    </div>
    <div class="lb-nav">
        <button class="nav-btn" id="navPrev" onclick="navLightbox(-1)">← Prev</button>
        <span class="nav-pos" id="navPos">1 / 1</span>
        <button class="nav-btn" id="navNext" onclick="navLightbox(1)">Next →</button>
    </div>
</div>

<script>
let allItems = [];
let filtered = [];
let currentFilter = 'all';
let currentIdx = 0;

async function loadGallery() {
    document.getElementById('gallery').innerHTML =
        '<div class="loading"><div class="spinner"></div><br>Loading captures...</div>';
    try {
        const res = await fetch('/api/gallery');
        allItems  = await res.json();
        applyFilter();
    } catch(e) {
        document.getElementById('gallery').innerHTML =
            '<div class="empty-state"><div class="icon">⚠</div><div class="msg">Failed to load gallery</div></div>';
    }
}

function setFilter(f) {
    currentFilter = f;
    ['all','detection','drop'].forEach(x => {
        const b = document.getElementById('f-'+x);
        b.className = 'filter-btn';
        if (x===f) b.classList.add('active', 'active-'+(f==='all'?'all':f==='detection'?'det':'drop'));
    });
    applyFilter();
}

function applyFilter() {
    if (currentFilter === 'all') {
        filtered = [...allItems];
    } else {
        filtered = allItems.filter(i => i.event === currentFilter);
    }
    applySort();
}

function applySort() {
    const s = document.getElementById('sortSelect').value;
    if (s === 'newest') filtered.sort((a,b) => b.timestamp.localeCompare(a.timestamp));
    if (s === 'oldest') filtered.sort((a,b) => a.timestamp.localeCompare(b.timestamp));
    if (s === 'person') filtered.sort((a,b) => (a.person_id||0) - (b.person_id||0));
    renderGallery();
}

function renderGallery() {
    document.getElementById('totalCount').textContent = allItems.length + ' photo' + (allItems.length===1?'':'s');
    const el = document.getElementById('gallery');
    if (!filtered.length) {
        el.innerHTML = '<div class="empty-state">'
            + '<div class="icon">📷</div>'
            + '<div class="msg">No photos yet</div>'
            + '<div class="sub">Photos appear here as persons are detected and drops are made</div>'
            + '</div>';
        return;
    }
    el.innerHTML = filtered.map((item, idx) => {
        const isDetect = item.event === 'detection';
        const hasGps   = item.gps_lat != null && item.gps_lon != null;
        const hasTh    = item.thermo_max_c != null;
        const gpsText  = hasGps ? item.gps_lat.toFixed(5)+', '+item.gps_lon.toFixed(5) : 'GPS pending';
        const thText   = hasTh  ? item.thermo_max_c.toFixed(1)+'°C max' : 'Thermal pending';
        const timeStr  = item.timestamp ? item.timestamp.split(' ')[1] : '—';
        const dateStr  = item.timestamp ? item.timestamp.split(' ')[0] : '—';
        return `
        <div class="img-card type-${item.event}" onclick="openLightbox(${idx})">
            <img class="img-thumb" src="${item.url}" alt="${item.label}" loading="lazy"
                 onerror="this.style.background='#0d1a24';this.alt='No image'">
            <div class="img-info">
                <div class="img-header">
                    <span class="img-label ${item.event}">${item.label}</span>
                    <span class="event-badge badge-${item.event}">${isDetect?'DETECTED':'DROP'}</span>
                </div>
                <div class="img-time">${dateStr} · ${timeStr}</div>
                <div class="img-tags">
                    <span class="tag ${hasGps?'tag-gps-live':'tag-gps-empty'}">📍 ${gpsText}</span>
                    <span class="tag ${hasTh?'tag-thermo-live':'tag-thermo-empty'}">🌡 ${thText}</span>
                </div>
                <div class="img-file">${item.filename}</div>
            </div>
        </div>`;
    }).join('');
}

function openLightbox(idx) {
    currentIdx = idx;
    showLightboxItem();
    document.getElementById('lightbox').classList.add('open');
    document.addEventListener('keydown', lbKeyHandler);
}

function showLightboxItem() {
    const item = filtered[currentIdx];
    if (!item) return;

    document.getElementById('lbImg').src   = item.url;
    document.getElementById('lbLabel').textContent = item.label + ' — ' + (item.event==='drop'?'DROP CAPTURE':'DETECTION');
    document.getElementById('navPos').textContent  = (currentIdx+1)+' / '+filtered.length;
    document.getElementById('navPrev').disabled    = currentIdx === 0;
    document.getElementById('navNext').disabled    = currentIdx === filtered.length-1;

    const hasGps   = item.gps_lat  != null && item.gps_lon != null;
    const hasAlt   = item.gps_alt  != null;
    const hasTMax  = item.thermo_max_c  != null;
    const hasTMean = item.thermo_mean_c != null;

    document.getElementById('lbMeta').innerHTML = `
        <div class="meta-section">
            <div class="meta-title">Identity</div>
            <div class="meta-row"><span class="meta-key">Person</span>  <span class="meta-val">${item.label}</span></div>
            <div class="meta-row"><span class="meta-key">Event</span>   <span class="meta-val">${item.event==='drop'?'Payload Drop':'First Detection'}</span></div>
            <div class="meta-row"><span class="meta-key">Date</span>    <span class="meta-val">${item.timestamp.split(' ')[0] || '—'}</span></div>
            <div class="meta-row"><span class="meta-key">Time</span>    <span class="meta-val">${item.timestamp.split(' ')[1] || '—'}</span></div>
            <div class="meta-row"><span class="meta-key">File</span>    <span class="meta-val" style="font-size:9px">${item.filename}</span></div>
        </div>
        <div class="meta-section">
            <div class="meta-title">📍 GPS Location</div>
            <div class="meta-row">
                <span class="meta-key">Latitude</span>
                <span class="meta-val ${hasGps?'live':'pending'}">${hasGps?item.gps_lat.toFixed(6):'pending hardware'}</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Longitude</span>
                <span class="meta-val ${hasGps?'live':'pending'}">${hasGps?item.gps_lon.toFixed(6):'pending hardware'}</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Altitude</span>
                <span class="meta-val ${hasAlt?'live':'pending'}">${hasAlt?item.gps_alt.toFixed(1)+' m':'pending hardware'}</span>
            </div>
        </div>
        <div class="meta-section">
            <div class="meta-title">🌡 Thermal</div>
            <div class="meta-row">
                <span class="meta-key">Max temp</span>
                <span class="meta-val ${hasTMax?'thermo':'pending'}">${hasTMax?item.thermo_max_c.toFixed(1)+' °C':'pending hardware'}</span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Mean temp</span>
                <span class="meta-val ${hasTMean?'thermo':'pending'}">${hasTMean?item.thermo_mean_c.toFixed(1)+' °C':'pending hardware'}</span>
            </div>
        </div>
        <div class="meta-section">
            <div class="meta-title">Download</div>
            <div class="meta-row">
                <span class="meta-key">Image</span>
                <span class="meta-val"><a href="${item.url}" download style="color:var(--accent)">↓ Download</a></span>
            </div>
            <div class="meta-row">
                <span class="meta-key">Folder</span>
                <span class="meta-val" style="color:var(--dim)">/${item.folder}/</span>
            </div>
        </div>
    `;
}

function navLightbox(dir) {
    const next = currentIdx + dir;
    if (next >= 0 && next < filtered.length) {
        currentIdx = next;
        showLightboxItem();
    }
}

function closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
    document.removeEventListener('keydown', lbKeyHandler);
}

function closeLightboxOutside(e) {
    if (e.target === document.getElementById('lightbox')) closeLightbox();
}

function lbKeyHandler(e) {
    if (e.key === 'ArrowLeft')  navLightbox(-1);
    if (e.key === 'ArrowRight') navLightbox(1);
    if (e.key === 'Escape')     closeLightbox();
}

// Auto-refresh every 10s
loadGallery();
setInterval(loadGallery, 10000);
</script>
</body>
</html>
"""

@app.route('/gallery')
def gallery():
    return render_template_string(GALLERY_PAGE)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HawkEye — Rescue Drone Module")
    print(f"  Captures     → {CAPTURES_DIR.resolve()}")
    print(f"  Heatmaps     → {HEATMAP_DIR.resolve()}")
    print(f"  Thermal HW   → {'YES' if _THERMAL_OK else 'NO (stub mode)'}")
    print(f"  Servo HW     → {'YES' if _DROPPER_OK else 'NO (stub mode)'}")
    print(f"  GPS port     → {GPS_PORT}")
    print("  Open: http://medidrone.local:5000")
    print("="*50 + "\n")
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        video_stream.stop()
