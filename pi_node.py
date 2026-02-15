#!/usr/bin/env python3
"""
pi_node.py
Raspberry Pi:
- Picamera2 -> YOLO -> annotated live stream (MJPEG)
- MLX90640 heatmap capture on event
- Save screenshot + heatmap files
- Upload both images + metadata to MongoDB using GridFS
"""

import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

# Pi camera + thermal
from picamera2 import Picamera2
import board
import busio
import adafruit_mlx90640

# MongoDB
from pymongo import MongoClient
import gridfs


# ---------------------------
# Config
# ---------------------------
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
IMGSZ = int(os.environ.get("IMGSZ", "320"))

SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", "Screenshot-dump")
HEATMAP_DIR = os.environ.get("HEATMAP_DIR", "heatmap_images")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

COOLDOWN_SECONDS = float(os.environ.get("COOLDOWN_SECONDS", "10"))
BOTTOM_THIRD_THRESHOLD = float(os.environ.get("BOTTOM_THIRD_THRESHOLD", "0.60"))

LOCATION_REFRESH_SECONDS = int(os.environ.get("LOCATION_REFRESH_SECONDS", "300"))
IPINFO_URL = os.environ.get("IPINFO_URL", "https://ipinfo.io/json")

# Mongo
MONGO_URI = "mongodb+srv://hawkeye:1928@m0.qolscs9.mongodb.net/?appName=M0"
MONGO_DB = "hawkeye"
MONGO_SOURCE = "pi"


# Stream server (Pi)
STREAM_HOST = os.environ.get("STREAM_HOST", "0.0.0.0")
STREAM_PORT = int(os.environ.get("STREAM_PORT", "5001"))

# Optional: turn on a local preview window on Pi desktop
LOCAL_PREVIEW = os.environ.get("LOCAL_PREVIEW", "0") == "1"
WIN_NAME = "HawkEye Pi Node"


# ---------------------------
# Location caching
# ---------------------------
def get_location_info_cached(state: dict) -> dict:
    """Fetch & cache approximate location from ipinfo.io every LOCATION_REFRESH_SECONDS."""
    now = time.time()

    # return cached if still fresh
    if state.get("location") and (now - state.get("location_ts", 0) < LOCATION_REFRESH_SECONDS):
        return state["location"]

    try:
        r = requests.get(IPINFO_URL, timeout=3)
        r.raise_for_status()
        data = r.json()

        loc = data.get("loc")  # "lat,lon"
        city = data.get("city")
        region = data.get("region")
        country = data.get("country")

        if loc and "," in loc:
            lat, lon = loc.split(",", 1)
            location = {
                "latitude": float(lat.strip()),
                "longitude": float(lon.strip()),
                "city": city,
                "region": region,
                "country": country,
                "provider": "ipinfo",
            }
        else:
            location = {"error": "Could not determine location", "provider": "ipinfo"}

    except Exception as e:
        location = {"error": f"Location error: {e}", "provider": "ipinfo"}

    state["location"] = location
    state["location_ts"] = now
    return location


def is_in_bottom_third_xyxy(bbox_xyxy, frame_height: int) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    return y2 > (frame_height * BOTTOM_THIRD_THRESHOLD)


def format_location_text(loc: dict) -> str:
    if not loc:
        return "Location: (none)"
    if "error" in loc:
        return f"Location: {loc['error']}"
    return f"{loc.get('city','')}, {loc.get('region','')}, {loc.get('country','')}  |  {loc.get('latitude',''):.5f}, {loc.get('longitude',''):.5f}"


# ---------------------------
# Thermal setup
# ---------------------------
def init_thermal():
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    frame = np.zeros((24 * 32,), dtype=float)
    return mlx, frame


def save_heatmap_png(mlx, frame_buf: np.ndarray, photo_counter: int) -> str:
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            mlx.getFrame(frame_buf)
            data = np.reshape(frame_buf, (24, 32))
            data = np.fliplr(data)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_path = os.path.join(HEATMAP_DIR, f"heatmap_{photo_counter:04d}.png")

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(data, vmin=float(np.min(data)), vmax=float(np.max(data)))
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Temperature (°C)")
            ax.set_title(f"Thermal Heatmap - {ts}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            return out_path

        except (ValueError, RuntimeError) as e:
            print(f"[THERMAL] Read failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(0.05)

    raise RuntimeError("[THERMAL] Failed to read MLX90640 after retries.")


# ---------------------------
# Mongo helpers (GridFS)
# ---------------------------
def init_mongo():
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is not set. Export MONGO_URI before running.")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    fs = gridfs.GridFS(db)
    detections = db["detections"]
    return client, db, fs, detections


def gridfs_put_file(fs: gridfs.GridFS, path: str, content_type: str) -> str:
    with open(path, "rb") as f:
        fid = fs.put(
            f,
            filename=os.path.basename(path),
            contentType=content_type,
            uploadedAt=datetime.now(timezone.utc),
        )
    return str(fid)


def upload_detection(detections_col, fs, loc_info, dets, screenshot_path, heatmap_path):
    screenshot_id = gridfs_put_file(fs, screenshot_path, "image/jpeg") if screenshot_path else None
    heatmap_id = gridfs_put_file(fs, heatmap_path, "image/png") if heatmap_path else None

    doc = {
        "ts": datetime.now(timezone.utc),
        "source": MONGO_SOURCE,
        "location": loc_info,
        "detections": dets,
        "screenshot_file_id": screenshot_id,
        "heatmap_file_id": heatmap_id,
        "screenshot_filename": os.path.basename(screenshot_path) if screenshot_path else None,
        "heatmap_filename": os.path.basename(heatmap_path) if heatmap_path else None,
    }
    detections_col.insert_one(doc)
    print("[MONGO] Inserted detection doc")


# ---------------------------
# MJPEG streaming server (Pi)
# ---------------------------
from flask import Flask, Response, jsonify

app = Flask(__name__)

latest_meta = {
    "ts": None,
    "location": None,
    "detections": [],
    "last_saved": None,
}


def make_stream_generator(model, picam2, mlx, thermal_frame, detections_col, fs):
    state = {
        "last_capture_time": 0.0,
        "photo_counter": 1,
        "location": None,
        "location_ts": 0.0,
    }

    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]

        results = model(frame, imgsz=IMGSZ)
        annotated = results[0].plot()

        # location overlay (cached)
        loc_info = get_location_info_cached(state)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cv2.putText(annotated, format_location_text(loc_info), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Time: {now_str}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        # gather detections
        dets = []
        person_in_zone = False

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for det in results[0].boxes.data:
                # [x1, y1, x2, y2, conf, cls]
                cls_id = int(det[5])
                conf = float(det[4])
                if cls_id != 0:
                    continue

                bbox = det[:4].tolist()
                dets.append({
                    "cls": "person",
                    "conf": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                })

                if is_in_bottom_third_xyxy(bbox, h):
                    person_in_zone = True

        # update latest meta (for /status)
        latest_meta["ts"] = datetime.now(timezone.utc).isoformat()
        latest_meta["location"] = loc_info
        latest_meta["detections"] = dets

        # cooldown capture
        screenshot_path = None
        heatmap_path = None
        if person_in_zone:
            now = time.time()
            if now - state["last_capture_time"] >= COOLDOWN_SECONDS:
                ts_ticks = cv2.getTickCount()
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"photo_{ts_ticks}.jpg")
                cv2.imwrite(screenshot_path, annotated)
                print(f"[CAPTURE] Saved photo: {screenshot_path}")

                try:
                    heatmap_path = save_heatmap_png(mlx, thermal_frame, state["photo_counter"])
                    print(f"[THERMAL] Saved heatmap: {heatmap_path}")
                except Exception as e:
                    print(f"[THERMAL] Heatmap save error: {e}")

                # upload to Mongo
                try:
                    upload_detection(detections_col, fs, loc_info, dets, screenshot_path, heatmap_path)
                    latest_meta["last_saved"] = {
                        "screenshot_path": screenshot_path,
                        "heatmap_path": heatmap_path,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as e:
                    print(f"[MONGO] Upload error: {e}")

                state["photo_counter"] += 1
                state["last_capture_time"] = now

        # optional local preview (Pi desktop)
        if LOCAL_PREVIEW:
            cv2.imshow(WIN_NAME, annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        # encode MJPEG frame
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.get("/video")
def video():
    # generator is attached in main() via app.config
    return Response(app.config["STREAM_GEN"], mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/status")
def status():
    return jsonify(latest_meta)


def main():
    print("[INIT] Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)

    print("[INIT] Starting Pi Camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888"}))
    picam2.start()

    print("[INIT] Starting thermal sensor...")
    mlx, thermal_frame = init_thermal()

    print("[INIT] Connecting to MongoDB...")
    client, db, fs, detections_col = init_mongo()
    print("[INIT] MongoDB connected")

    if LOCAL_PREVIEW:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    # attach generator
    app.config["STREAM_GEN"] = make_stream_generator(model, picam2, mlx, thermal_frame, detections_col, fs)

    try:
        print(f"[RUN] Streaming on http://<PI_IP>:{STREAM_PORT}/video")
        app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, threaded=True)
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
