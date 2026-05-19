# gunicorn.conf.py  — loaded automatically by gunicorn
import threading, os

def post_fork(server, worker):
    """
    Re-start the model download thread in each worker.
    Threads do not survive os.fork() so the master's download thread
    is gone by the time workers run. This hook restarts it.
    If the model file already exists the download function returns
    immediately, so this is cheap on subsequent deploys.
    """
    # Import the app-level globals and re-trigger download
    try:
        import app as _app
        _app._MODEL_READY = threading.Event()
        _app._MODEL_ERROR  = None
        t = threading.Thread(
            target=_app._download_model, daemon=True, name="model-dl"
        )
        t.start()
    except Exception as e:
        server.log.warning(f"[VERITY] post_fork hook error: {e}")
