FROM python:3.11-slim

# ── System packages (Debian Bookworm names) ────────────────────────────────────
# libgles2   → provides libGLESv2.so.2  (required by MediaPipe tasks API)
# libgl1     → provides libGL.so.1      (OpenCV / MediaPipe)
# libegl1    → provides libEGL.so.1     (MediaPipe GPU init path)
# The rest are standard OpenCV headless dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgles2 \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (gunicorn is already in requirements.txt) ──────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake the MediaPipe model into the image ────────────────────────────────────
# Downloading here means zero startup delay and no runtime dependency
# on Google Storage being reachable.
COPY download_model.py .
RUN python download_model.py

# ── Application ────────────────────────────────────────────────────────────────
COPY . .

# ── Runtime ────────────────────────────────────────────────────────────────────
# Shell form so $PORT is expanded (Render injects it as an env variable).
# Single worker: free tier has 512 MB RAM; MediaPipe model uses ~150 MB.
# Timeout 120 s covers face landmarker init on the very first request.
EXPOSE 8080
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --timeout 120 \
    --worker-class sync
