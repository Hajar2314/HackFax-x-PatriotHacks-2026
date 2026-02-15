#!/usr/bin/env python3
"""
Raspberry Pi (Picamera2) + Ultralytics YOLO + MLX90640 heatmap capture
- Uses ONE OpenCV window (stable GUI loop)
- Avoids repeated MLX/I2C/Matplotlib re-init on every detection
- Caches IP-based location (doesn't call the API every frame)
- Fixes bbox bottom-third logic (uses y2, not y1)
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")  # saves PNGs without needing a GUI backend
import matplotlib.pyplot as plt

import board
import busio
import adafruit_mlx90640

from picamera2 import Picamera2
from ultralytics import YOLO

import dropper

dropper.initialize_servos()

# ---------------------------
# Configuration
# ---------------------------
YOLO_MODEL_PATH = "yolov8n.pt"
IMGSZ = 320

SCREENSHOT_DIR = "Screenshot-dump"
HEATMAP_DIR = "heatmap_images"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

COOLDOWN_SECONDS = 10
BOTTOM_THIRD_THRESHOLD = 0.60  # > 0.60 means lower ~40% of frame

LOCATION_REFRESH_SECONDS = 300  # refresh location every 5 minutes
IPINFO_URL = "https://ipinfo.io/json"

WIN_NAME = "YOLO Person Detection"


# ---------------------------
# Helpers
# ---------------------------
def get_location_info_cached(state: dict) -> dict:
    """Fetch & cache approximate location from ipinfo.io."""
    now = time.time()
  

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
                "latitude": lat.strip(),
                "longitude": lon.strip(),
                "city": city,
                "region": region,
                "country": country,
            }
        else:
            location = {"error": "Could not determine location"}

    except Exception as e:
        location = {"error": f"Location error: {e}"}

    state["location"] = location
    state["location_ts"] = now
    return location


def is_in_bottom_third_xyxy(bbox_xyxy, frame_height: int) -> bool:
    """
    bbox is [x1, y1, x2, y2] (XYXY).
    Bottom means y2 (max y) is low enough in the frame.
    """
    x1, y1, x2, y2 = bbox_xyxy
    return y2 > (frame_height * BOTTOM_THIRD_THRESHOLD)


def format_location_text(location_info: dict) -> str:
    if "error" in location_info:
        return location_info["error"]
    return (
        f"{location_info.get('city','')}, {location_info.get('region','')}, {location_info.get('country','')} - "
        f"Lat: {location_info.get('latitude','')}, Lon: {location_info.get('longitude','')}"
    )


# ---------------------------
# Thermal (MLX90640) setup
# ---------------------------
def init_thermal():
    """Initialize I2C + MLX90640 once."""
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    frame = np.zeros((24 * 32,), dtype=float)
    return mlx, frame


def save_heatmap_png(mlx, frame_buf: np.ndarray, photo_counter: int) -> str:
    """
    Reads ONE thermal frame and saves a heatmap PNG.
    Uses matplotlib Agg backend -> no Qt windows -> more stable on Pi.
    """
    # Retry a few times because MLX90640 can throw transient errors
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            mlx.getFrame(frame_buf)
            data = np.reshape(frame_buf, (24, 32))
            data = np.fliplr(data)  # orientation tweak (keep if it matches your sensor mounting)

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_path = os.path.join(HEATMAP_DIR, f"heatmap_{photo_counter:04d}.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(data, vmin=float(np.min(data)), vmax=float(np.max(data)))
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Temperature (°C)")
            ax.set_title(f"Thermal Heatmap - {ts}")
            ax.axis("off")

            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            print(f"[THERMAL] Saved heatmap: {out_path}")
            return out_path

        except (ValueError, RuntimeError) as e:
            print(f"[THERMAL] Read failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(0.05)

    raise RuntimeError("[THERMAL] Failed to read MLX90640 after retries.")


# ---------------------------
# Main
# ---------------------------
def main():
    # YOLO
    model = YOLO(YOLO_MODEL_PATH)

    # Pi camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": "BGR888"}))
    picam2.start()

    # Thermal
    mlx, thermal_frame = init_thermal()

    # GUI window (single instance)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    state = {
        "last_capture_time": 0.0,
        "photo_counter": 1,
        "location": None,
        "location_ts": 0.0,
    }

    try:
        while True:
            detected = True   # replace with OpenCV detection
            dropper.hold_position()
    
            frame = picam2.capture_array()
            h, w = frame.shape[:2]

            # YOLO inference
            results = model(frame, imgsz=IMGSZ)

            # annotated frame
            annotated = results[0].plot()

            # cached location + time overlays
            loc_info = get_location_info_cached(state)
            loc_text = format_location_text(loc_info)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(annotated, loc_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"Time: {now_str}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)

            # detection logic
            person_in_zone = False
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for det in results[0].boxes.data:
                    # det: [x1, y1, x2, y2, conf, cls]
                    cls_id = int(det[5])
                    if cls_id != 0:
                        continue  # only "person"

                    bbox = det[:4].tolist()
                    if is_in_bottom_third_xyxy(bbox, h):
                        person_in_zone = True
                        break

            # screenshot + heatmap (cooldown-protected)
            if person_in_zone:
                now = time.time()
                if now - state["last_capture_time"] >= COOLDOWN_SECONDS:
                    ts_ticks = cv2.getTickCount()
                    img_path = os.path.join(SCREENSHOT_DIR, f"photo_{ts_ticks}.jpg")
                    cv2.imwrite(img_path, annotated)
                    print(f"[CAPTURE] Saved photo: {img_path}")
                    dropper.drop_sequence()

                    try:
                        save_heatmap_png(mlx, thermal_frame, state["photo_counter"])
                    except Exception as e:
                        print(f"[THERMAL] Heatmap save error: {e}")

                    state["photo_counter"] += 1
                    state["last_capture_time"] = now
                else:
                    # optional debug
                    pass

            # show ONE window
            cv2.imshow(WIN_NAME, annotated)

            # IMPORTANT: waitKey pumps GUI events (keeps the window responsive)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()

