# gunicorn.conf.py
# -----------------------------------------------------------------
# preload_app = False  (default) — each worker imports the app
# independently, which is required for MediaPipe/TFLite fork safety.
#
# If your platform forces --preload via CLI, this file overrides it.
# -----------------------------------------------------------------
preload_app = False
workers = 2
worker_class = "sync"
timeout = 120
bind = "0.0.0.0:10000"
