#!/usr/bin/env python3
"""
Pre-download the MediaPipe face landmarker model.
Run this before starting gunicorn so the model is ready immediately
on the first request. Idempotent — safe to run multiple times.
"""
import os, sys, urllib.request, time

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# App directory — same path app.py will look for the model
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "face_landmarker.task")

def download():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"[VERITY] Model already present ({size_mb:.1f} MB) — skipping download.")
        return

    print(f"[VERITY] Downloading face landmarker model…", flush=True)
    t0  = time.time()
    tmp = MODEL_PATH + ".tmp"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        os.replace(tmp, MODEL_PATH)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print(f"[VERITY] Model ready: {MODEL_PATH} ({size_mb:.1f} MB, {elapsed:.1f}s)", flush=True)
    except Exception as e:
        if os.path.exists(tmp):
            os.unlink(tmp)
        print(f"[VERITY] Download failed: {e}", file=sys.stderr)
        sys.exit(1)          # fail loudly so Render stops the deploy

if __name__ == "__main__":
    download()
