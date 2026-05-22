"""
VERITY — Deception Analysis System  v2.0
Based on Paul Ekman & Wallace V. Friesen's "Unmasking the Face" (2003)

Changes from v1:
  [1] Thread-local MediaPipe instances  — tasks API (mp.solutions removed ≥0.10.9)
  [2] Per-session isolated state        — no shared globals between users
  [3] Session TTL cleanup               — idle sessions purged automatically
  [4] sklearn MLP emotion classifier    — synthetic-data bootstrapped; drop-in
                                          replaceable with real labelled data
  [5] scipy peak-detection micro-exprs  — replaces noisy frame-diff heuristic
  [6] Exponential moving-average smoother — eliminates per-frame score jitter
"""

import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import base64
import time
import os
import uuid
import tempfile
import threading
import warnings
from collections import deque

from flask import Flask, render_template, request, jsonify, session
from classifier import classify_emotion_frame
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Startup error log — appended during import so /debug can surface them
_STARTUP_ERRORS: list[str] = []
_STARTUP_LOG:   list[str] = []

def _log(msg: str) -> None:
    print(msg, flush=True)
    _STARTUP_LOG.append(msg)

app = Flask(__name__)
_secret = os.environ.get("VERITY_SECRET")
if not _secret:
    # Stable fallback for single-worker local dev only.
    # On any real deployment set VERITY_SECRET=<random hex string> in env vars.
    import hashlib, socket
    _secret = hashlib.sha256(socket.gethostname().encode()).hexdigest()
app.secret_key = _secret


# ══════════════════════════════════════════════════════════════
# [1] THREAD-LOCAL MEDIAPIPE INSTANCES  (mediapipe.tasks API)
#     mp.solutions was removed in mediapipe ≥ 0.10.9.
#     We now use the stable mediapipe.tasks.python.vision API.
#     The model file is downloaded once to /tmp on first boot.
# ══════════════════════════════════════════════════════════════

from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
# Use app directory so download_model.py and app.py share the same file.
# Fall back to /tmp if the directory isn't writable (e.g. read-only FS).
_APP_DIR    = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = (
    os.path.join(_APP_DIR, "face_landmarker.task")
    if os.access(_APP_DIR, os.W_OK)
    else os.path.join(tempfile.gettempdir(), "face_landmarker.task")
)

# ── Synchronous lazy model initialisation ────────────────────
# The model is downloaded on the first face-detection call.
# An fcntl exclusive lock ensures only one worker downloads at a time;
# others wait and then find the file already present.
# Gunicorn timeout is 120 s — a ~30 MB download fits comfortably.
# No threads, no Events, no post_fork hooks needed.

_thread_local = threading.local()


def _ensure_model() -> None:
    """
    Fallback model download — only runs if download_model.py was not
    executed before gunicorn started (local dev, non-Render deploys).
    On Render the model is pre-downloaded by the Procfile start command.
    """
    if os.path.exists(_MODEL_PATH):
        return
    print("[VERITY] WARNING: model not pre-downloaded; downloading now "
          "(first request will be slow). Run download_model.py first.", flush=True)
    try:
        import fcntl
        lock_path = _MODEL_PATH + ".lock"
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if not os.path.exists(_MODEL_PATH):
                    tmp = _MODEL_PATH + ".tmp"
                    urllib.request.urlretrieve(_MODEL_URL, tmp)
                    os.replace(tmp, _MODEL_PATH)
                    print("[VERITY] Model ready:", _MODEL_PATH, flush=True)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except ImportError:
        # fcntl unavailable (Windows) — download without lock
        if not os.path.exists(_MODEL_PATH):
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def _make_landmarker_opts() -> "_mp_vision.FaceLandmarkerOptions":
    return _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )


def get_face_landmarker() -> "_mp_vision.FaceLandmarker":
    """
    Return a thread-local FaceLandmarker instance.
    Downloads the model file on first call if not already present.
    """
    if not hasattr(_thread_local, "landmarker"):
        _ensure_model()                          # blocks ~5-30 s on first call only
        _thread_local.landmarker = _mp_vision.FaceLandmarker.create_from_options(
            _make_landmarker_opts()
        )
    return _thread_local.landmarker


# ══════════════════════════════════════════════════════════════
# Ekman landmark regions  (MediaPipe 468-point mesh)
# ══════════════════════════════════════════════════════════════

LANDMARKS = {
    "inner_brow_left":    [107, 66, 105, 63, 70],
    "inner_brow_right":   [336, 296, 334, 293, 300],
    "outer_brow_left":    [46, 53, 52, 65, 55],
    "outer_brow_right":   [285, 295, 282, 283, 276],
    "upper_lid_left":     [159, 158, 157, 173, 133],
    "upper_lid_right":    [386, 385, 384, 398, 362],
    "lower_lid_left":     [145, 144, 163, 7],
    "lower_lid_right":    [374, 373, 390, 249],
    "cheek_left":         [116, 117, 118, 119, 120],
    "cheek_right":        [345, 346, 347, 348, 349],
    "mouth_corner_left":  [61, 146, 91, 181, 84],
    "mouth_corner_right": [291, 375, 321, 405, 314],
    "upper_lip":          [13, 312, 311, 310, 415, 308, 78, 191, 80, 81, 82],
    "lower_lip":          [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13],
    "jaw":                [152, 377, 400, 378, 379, 365, 397, 288],
    "nose_bridge":        [6, 197, 195, 5],
    "nose_lower":         [4, 240, 98, 97, 2, 326, 327, 460],
}

# Canonical feature order (must stay fixed — classifier was trained on this)
FEATURE_NAMES = [
    "inner_brow_raise", "outer_brow_raise", "brow_gap",
    "eye_aperture",     "lower_lid_tension", "cheek_raise",
    "mouth_width",      "mouth_open",        "lip_corner_dir",
    "upper_lip_raise",  "lip_press",         "nose_wrinkle",
    "jaw_drop",
]

EMOTION_LABELS = ["happiness", "surprise", "fear", "anger", "disgust", "sadness"]


# ══════════════════════════════════════════════════════════════
# [4] SKLEARN MLP EMOTION CLASSIFIER
#
#     Architecture: StandardScaler → MLP(64→32) → softmax
#
#     Bootstrap strategy: synthetic Gaussian clusters centred on
#     each emotion's canonical FACS action-unit pattern.
#     The pipeline is fully serialisable — swap in a real dataset
#     (CK+, AffectNet, AffWild2) by calling clf.fit(X_real, y_real).
# ══════════════════════════════════════════════════════════════

def _synthetic_training_data(n_per_class: int = 600, seed: int = 42) -> tuple:
    """
    Generate labelled samples from Ekman's AU descriptions.

    Feature vector (FEATURE_NAMES order):
      0  inner_brow_raise   4  lower_lid_tension   8  lip_corner_dir
      1  outer_brow_raise   5  cheek_raise         9  upper_lip_raise
      2  brow_gap           6  mouth_width        10  lip_press
      3  eye_aperture       7  mouth_open         11  nose_wrinkle
                                                  12  jaw_drop
    """
    rng = np.random.default_rng(seed)

    def cluster(mu, n=n_per_class, sig=0.013):
        return np.clip(rng.normal(mu, sig, (n, len(FEATURE_NAMES))), -0.5, 1.0)

    # ── Canonical AU centroids per emotion ───────────────
    # Happiness  AU6+12: cheek raise + lip corner pull + lower lid crinkle
    H  = cluster([0.000, 0.005, 0.40, 0.25, 0.25, 0.055, 0.50, 0.05,  0.08, 0.08, 0.50, 0.05, 0.02])

    # Surprise   AU1+2+5+26: inner+outer brow raise, wide eye, jaw drop
    Su = cluster([0.030, 0.040, 0.48, 0.36, 0.12, 0.010, 0.45, 0.18,  0.00, 0.07, 0.30, 0.04, 0.07])

    # Fear       AU1+4+5+7+20+26: inner brow raise+draw, wide eye, lip stretch
    Fe = cluster([0.040, 0.010, 0.33, 0.33, 0.14, 0.000, 0.49, 0.14,  0.03, 0.09, 0.35, 0.04, 0.05])

    # Anger      AU4+5+7+23+24: brow draw, tense lids, lip press
    An = cluster([0.005, 0.005, 0.27, 0.22, 0.29, 0.000, 0.37, 0.03, -0.01, 0.07, 0.76, 0.06, 0.01])

    # Disgust    AU9+15+16+17: nose wrinkle, upper lip raise, corner down
    Di = cluster([0.005, 0.005, 0.38, 0.20, 0.15, 0.000, 0.36, 0.06, -0.04, 0.14, 0.55, 0.13, 0.01])

    # Sadness    AU1+4+15+17: inner brow raise+draw, lip corners down
    Sa = cluster([0.020, 0.002, 0.31, 0.20, 0.10, 0.000, 0.38, 0.04, -0.06, 0.07, 0.55, 0.04, 0.01])

    X = np.vstack([H, Su, Fe, An, Di, Sa])
    y = np.repeat(np.arange(6), n_per_class)
    return X, y


def _build_classifier() -> Pipeline:
    print("  [VERITY] Building emotion classifier (synthetic bootstrap)…", flush=True)
    X, y = _synthetic_training_data()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False,
        )),
    ])
    pipe.fit(X, y)
    print("  [VERITY] Classifier ready.", flush=True)
    return pipe


_emotion_clf      = None
_emotion_clf_lock = threading.Lock()

def _get_classifier():
    global _emotion_clf
    if _emotion_clf is None:
        with _emotion_clf_lock:
            if _emotion_clf is None:
                _emotion_clf = _build_classifier()
    return _emotion_clf


def classify_emotion(f: dict) -> dict:
    """
    Ekman AU emotion classifier — uses trained MLP when available,
    falls back to rule-based Ekman AU mapping.

    f must be ADJUSTED (delta) features: current - baseline.
    f contains (current_feature - baseline_feature) values, so:
      • Neutral face → all deltas ≈ 0 → all scores ≈ 0 → ~1/6 each after floor
      • Genuine smile → lip_corner_dir and cheek_raise deltas positive → happiness rises
      • Surprise     → outer_brow_raise, eye_aperture, jaw_drop deltas positive
      • etc.

    max(delta, 0) means only elevations ABOVE the person's own neutral count.
    Negative deltas (features dropping below neutral) feed sadness/disgust only.
    """
    # Use trained model if one has been saved via /train_model
    if _TRAINED_MODEL is not None:
        try:
            fv    = np.array([[f.get(k, 0.0) for k in FEATURE_NAMES]])
            probs = _TRAINED_MODEL.predict_proba(fv)[0]
            return {label: round(float(p), 3)
                    for label, p in zip(EMOTION_LABELS, probs)}
        except Exception:
            pass   # fall back to rule-based on any error

    scores: dict = {}

    # HAPPINESS  AU6+12: lip corner pull UP + cheek rise + lower lid crinkle
    scores["happiness"] = np.clip(
        0.40 * max(f["lip_corner_dir"],    0) +
        0.30 * max(f["cheek_raise"],       0) +
        0.30 * max(f["lower_lid_tension"], 0),
        0, 1)

    # SURPRISE   AU1+2+5+26: brow raise + eye wide + jaw drop (all elevated)
    scores["surprise"] = np.clip(
        0.35 * max(f["outer_brow_raise"], 0) +
        0.30 * max(f["eye_aperture"],     0) +
        0.35 * max(f["jaw_drop"],         0),
        0, 1)

    # FEAR       AU1+4+5+7+20+26: inner brow + wide eye + open/wide mouth
    scores["fear"] = np.clip(
        0.40 * max(f["inner_brow_raise"], 0) +
        0.25 * max(f["eye_aperture"],     0) +
        0.20 * max(f["mouth_open"],       0) +
        0.15 * max(f["mouth_width"],      0),
        0, 1)

    # ANGER      AU4+7+23+24: brow NARROWS (negative brow_gap delta) + lid + press
    brow_draw = max(-f["brow_gap"], 0) * 2   # negative delta → brows drawn together
    scores["anger"] = np.clip(
        0.35 * brow_draw +
        0.30 * max(f["lower_lid_tension"], 0) +
        0.25 * max(f["lip_press"],         0) +
        0.10 * max(f["eye_aperture"],      0),
        0, 1)

    # DISGUST    AU9+15+16: upper lip raise + nose wrinkle + corners DOWN
    scores["disgust"] = np.clip(
        0.45 * max(f["upper_lip_raise"],  0) +
        0.35 * max(f["nose_wrinkle"],     0) +
        0.20 * max(-f["lip_corner_dir"],  0),
        0, 1)

    # SADNESS    AU1+4+15+17: inner brow + corners DOWN + cheek DROP
    scores["sadness"] = np.clip(
        0.45 * max(f["inner_brow_raise"],  0) +
        0.30 * max(-f["lip_corner_dir"],   0) +
        0.25 * max(-f["cheek_raise"],      0),
        0, 1)

    # Floor: ensures neutral face returns ~1/6 each rather than 0/0
    for k in scores:
        scores[k] = max(scores[k], 0.008)

    total = sum(scores.values())
    return {k: round(float(v / total), 3) for k, v in scores.items()}




# ══════════════════════════════════════════════════════════════
# [6] EXPONENTIAL MOVING-AVERAGE SMOOTHER
#     Eliminates per-frame deception score jitter.
#     alpha=0.15 → ~6-frame (200 ms) effective averaging window.
# ══════════════════════════════════════════════════════════════

class ExponentialSmoother:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return round(float(self._value), 3)

    def reset(self):
        self._value = None


# ══════════════════════════════════════════════════════════════
# [2] & [3] PER-SESSION STATE  +  TTL CLEANUP
#     Every browser session gets its own isolated deque buffers,
#     baseline, frame counter, and smoother — no global mutation.
#     Sessions idle > SESSION_TTL seconds are garbage-collected.
# ══════════════════════════════════════════════════════════════

HISTORY_LEN  = 90        # ~3 s at 30 fps
SESSION_TTL  = 3600      # 1 hour idle → purge

_session_store: dict = {}
_session_lock         = threading.Lock()


def _purge_stale_sessions():
    cutoff = time.time() - SESSION_TTL
    stale  = [sid for sid, s in _session_store.items()
               if s["last_access"] < cutoff]
    for sid in stale:
        del _session_store[sid]


def get_session_state(sid: str) -> dict:
    """Retrieve (or create) the mutable state dict for a session."""
    with _session_lock:
        if sid not in _session_store:
            _session_store[sid] = {
                "history":       deque(maxlen=HISTORY_LEN),
                "timestamps":    deque(maxlen=HISTORY_LEN),
                "baseline":      {},
                "frame_count":   0,
                "start_time":    time.time(),
                "smoother":      ExponentialSmoother(alpha=0.15),
                "last_access":   time.time(),
                # CNN async fields — CNN never blocks the request thread
                "last_emotions": {e: round(1/6, 3) for e in EMOTION_LABELS},
                "cnn_running":   False,
            }
        state = _session_store[sid]
        state["last_access"] = time.time()
        if len(_session_store) % 100 == 0:
            _purge_stale_sessions()
        return state


def _run_cnn_async(state: dict, face_crop) -> None:
    """
    Run classify_emotion_frame in a daemon thread so it never stalls
    the /analyse response.  Result is written back to session state;
    the next frame picks it up automatically.
    """
    try:
        result = classify_emotion_frame(face_crop)
        state["last_emotions"] = result
    except Exception as exc:
        print(f"[VERITY] CNN inference error: {exc}", flush=True)
    finally:
        state["cnn_running"] = False


# ══════════════════════════════════════════════════════════════
# Helper geometry  (unchanged)
# ══════════════════════════════════════════════════════════════

def lm(landmarks, idx, w, h):
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h])


def region_spread(landmarks, indices, w, h):
    pts = np.array([lm(landmarks, i, w, h) for i in indices])
    c   = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def angle_of_lift(landmarks, tip_idx, base_idx, w, h):
    tip  = lm(landmarks, tip_idx,  w, h)
    base = lm(landmarks, base_idx, w, h)
    return float(base[1] - tip[1])   # image Y inverted → positive = upward lift


# ══════════════════════════════════════════════════════════════
# Feature extraction  (13 normalised scalars)
# ══════════════════════════════════════════════════════════════

def extract_features(landmarks, w: int, h: int) -> dict:
    """
    Compute FACS-aligned scalar features from a MediaPipe landmark set.
    All distances normalised by inter-ocular distance (IOD).
    """
    left_eye_outer  = lm(landmarks, 33,  w, h)
    right_eye_outer = lm(landmarks, 263, w, h)
    iod = max(euclidean(left_eye_outer, right_eye_outer), 1.0)

    def norm(d): return d / iod

    # ── Brow / Forehead ──────────────────────────────────
    # AU1: inner brow raise  — fear/sad; hardest to voluntarily produce (Ekman p.148)
    inner_brow_raise = norm((
        angle_of_lift(landmarks, 107, 33,  w, h) +
        angle_of_lift(landmarks, 336, 263, w, h)
    ) / 2)

    # AU4: brow lowerer / draw-together  — anger emblem (Ekman p.149)
    brow_gap = norm(euclidean(lm(landmarks, 107, w, h), lm(landmarks, 336, w, h)))

    # AU2: outer brow raise  — surprise emblem; easy to fake
    outer_brow_raise = norm((
        angle_of_lift(landmarks, 70,  33,  w, h) +
        angle_of_lift(landmarks, 300, 263, w, h)
    ) / 2)

    # ── Eyes ─────────────────────────────────────────────
    # AU5: upper lid raiser  — fear stare; surprise
    eye_aperture = (
        norm(euclidean(lm(landmarks, 159, w, h), lm(landmarks, 145, w, h))) +
        norm(euclidean(lm(landmarks, 386, w, h), lm(landmarks, 374, w, h)))
    ) / 2

    # AU7: lid tightener  — Duchenne happiness; anger
    lower_lid_tension = norm((
        region_spread(landmarks, LANDMARKS["lower_lid_left"],  w, h) +
        region_spread(landmarks, LANDMARKS["lower_lid_right"], w, h)
    ) / 2)

    # AU6: cheek raiser  — Duchenne marker; absent in fake smiles
    # Reference: mouth corners (61/291). Cheek landmarks (116/345) sit ABOVE
    # the corners, giving positive values that increase when cheeks rise.
    cheek_raise = norm((
        angle_of_lift(landmarks, 116, 61,  w, h) +
        angle_of_lift(landmarks, 345, 291, w, h)
    ) / 2)

    # ── Mouth / Lower face  (primary management zone; Ekman p.146) ────
    mouth_L = lm(landmarks, 61,  w, h)
    mouth_R = lm(landmarks, 291, w, h)

    # AU25: lips part; AU26: jaw drop
    mouth_width = norm(euclidean(mouth_L, mouth_R))
    mouth_open  = norm(euclidean(lm(landmarks, 13, w, h), lm(landmarks, 14, w, h)))

    # AU12: lip corner puller (happiness) / AU15: lip corner depressor (sadness)
    # Lower lip center (14) is the neutral-height reference for corner elevation:
    #   positive → corners above lower lip center (smile)
    #   negative → corners below lower lip center (frown)
    lower_lip_c_ref = lm(landmarks, 14, w, h)
    lip_corner_dir  = norm((
        float(lower_lip_c_ref[1] - mouth_L[1]) +
        float(lower_lip_c_ref[1] - mouth_R[1])
    ) / 2)

    # AU10: upper lip raiser  — disgust
    upper_lip_raise = norm(float(
        lm(landmarks, 2, w, h)[1] - lm(landmarks, 13, w, h)[1]
    ))

    # AU17+18: lip press / chin raiser  — anger
    lip_press = 1.0 - min(mouth_open * 10, 1.0)

    # AU9: nose wrinkler  — disgust
    nose_wrinkle = norm(region_spread(landmarks, LANDMARKS["nose_lower"], w, h))

    # AU26/27: jaw drop  — surprise / fear
    jaw_drop = norm(float(lm(landmarks, 152, w, h)[1] - lm(landmarks, 14, w, h)[1]))

    return {
        "inner_brow_raise":  round(inner_brow_raise,  4),
        "outer_brow_raise":  round(outer_brow_raise,  4),
        "brow_gap":          round(brow_gap,           4),
        "eye_aperture":      round(eye_aperture,       4),
        "lower_lid_tension": round(lower_lid_tension,  4),
        "cheek_raise":       round(cheek_raise,        4),
        "mouth_width":       round(mouth_width,        4),
        "mouth_open":        round(mouth_open,         4),
        "lip_corner_dir":    round(lip_corner_dir,     4),
        "upper_lip_raise":   round(upper_lip_raise,    4),
        "lip_press":         round(lip_press,          4),
        "nose_wrinkle":      round(nose_wrinkle,       4),
        "jaw_drop":          round(jaw_drop,           4),
    }


# ══════════════════════════════════════════════════════════════
# [5] SCIPY PEAK-DETECTION MICRO-EXPRESSION DETECTOR
#
#     v1 compared adjacent frames — extremely noisy.
#     v2 runs scipy.signal.find_peaks over a smoothed 90-frame
#     signal, requiring:
#       • prominence ≥ threshold*0.6  (stands above local baseline)
#       • width in [1, 5] frames      (1/30 s – 5/30 s ≈ Ekman range)
#       • minimum absolute height     (not just a relative blip)
#
#     Checked for 4 suppressed emotions (fear, anger, disgust, sadness).
# ══════════════════════════════════════════════════════════════

_MICRO_CHECKS = [
    # (emotion_label, feature_key, high_peak, abs_threshold)
    ("FEAR",    "inner_brow_raise", True,  0.015),
    ("ANGER",   "brow_gap",         False, 0.300),   # low = anger
    ("DISGUST", "upper_lip_raise",  True,  0.060),
    ("SADNESS", "inner_brow_raise", True,  0.012),
]


def detect_micro_expressions(history: list) -> list[tuple[str, int]]:
    """
    Scan the feature history for brief transient peaks indicative of
    suppressed micro-expressions.  Returns [(emotion, frame_index), …].

    Only the last 30 frames (~1 s at 30 fps) are scanned — scanning the
    full 90-frame deque with scipy every frame was a significant CPU cost.
    """
    # Clamp to most recent 30 frames — micro-expressions are brief by definition
    history = history[-30:]
    if len(history) < 15:
        return []

    results = []
    for emotion, key, high, thresh in _MICRO_CHECKS:
        signal = np.array([f.get(key, 0.0) for f in history], dtype=float)
        if not high:
            signal = -signal

        # 3-frame uniform smoothing to remove single-frame noise
        smoothed = uniform_filter1d(signal, size=3)

        peaks, _ = find_peaks(
            smoothed,
            prominence=thresh * 0.6,
            width=(1, 5),
            height=thresh,
        )
        for idx in peaks:
            # De-duplicate: skip if another detection within 5 frames
            if not any(abs(idx - prev_idx) < 5
                       for _, prev_idx in results if _ == emotion):
                results.append((emotion, int(idx)))

    return results


# ══════════════════════════════════════════════════════════════
# Deception analysis  (morphology · timing · micro · leakage)
# Source: "Unmasking the Face" Ch.11 pp.143–155
# ══════════════════════════════════════════════════════════════

def analyse_deception(
    features: dict,
    baseline: dict,
    emotions: dict,
    history_features: list,
    history_times: list,
    smoother: ExponentialSmoother,
    micro_exprs: list | None = None,
) -> dict:
    """
    Ekman deception analysis using raw features + calibrated baseline.

    All threshold comparisons use  (feature - baseline)  explicitly so
    the meaning is clear: "has this AU moved significantly from the
    person's own neutral?" rather than testing raw absolute values.
    """
    clues      = []
    components = []

    dominant      = max(emotions, key=emotions.get)
    dominant_conf = emotions[dominant]

    # Per-feature delta: how much each AU has moved from calibrated neutral.
    def d(key: str) -> float:
        return features.get(key, 0) - baseline.get(key, features.get(key, 0))

    # Semantic helpers — thresholds chosen from typical AU movement ranges
    def elevated(key: str, thresh: float = 0.012) -> bool:
        return d(key) > thresh            # meaningfully above neutral

    def absent(key: str, thresh: float = 0.010) -> bool:
        return d(key) <= thresh           # at or below neutral — AU not active

    def depressed(key: str, thresh: float = 0.010) -> bool:
        return d(key) < -thresh           # pulled below neutral

    # ── Expression intensity gate ─────────────────────────────
    # Skip deception analysis when the face is near-neutral.
    # Prevents noise accumulation during resting expressions.
    intensity = sum(abs(d(k)) for k in [
        "lip_corner_dir", "cheek_raise", "outer_brow_raise",
        "inner_brow_raise", "eye_aperture", "jaw_drop", "mouth_open",
    ])
    if intensity < 0.025:
        raw_score = 0.04 + float(np.random.uniform(0, 0.03))
        return {
            "deception_score":    smoother.update(raw_score),
            "raw_score":          round(raw_score, 3),
            "clues":              [],
            "dominant_emotion":   dominant,
            "emotion_confidence": round(dominant_conf, 3),
        }

    # ── 1. MORPHOLOGY ────────────────────────────────────────
    # Ekman p.146: lower face is managed; brow/forehead leaks true emotion.

    # Non-Duchenne smile: lip corners pulled up but neither Duchenne AU fires
    if emotions["happiness"] > 0.38 and elevated("lip_corner_dir"):
        if absent("cheek_raise") and absent("lower_lid_tension"):
            components.append(0.65)
            clues.append({
                "type": "MORPHOLOGY", "label": "Non-Duchenne Smile",
                "detail": ("Lip corners raised above baseline but AU6 cheek-raise and "
                           "AU7 lower-lid tension absent — Ekman's hallmark of a social "
                           "/ posed smile (p.105–120)."),
                "severity": "HIGH",
            })

    # Simulated surprise: brow raised but eyes not actually wider
    if emotions["surprise"] > 0.42 and elevated("outer_brow_raise"):
        if absent("eye_aperture", thresh=0.018):
            components.append(0.55)
            clues.append({
                "type": "MORPHOLOGY", "label": "Suppressed Eyelid Opening",
                "detail": ("Outer brow elevated above baseline (AU2 — easy emblem) but "
                           "eye aperture not wider than neutral; Ekman: the only deception "
                           "clue in simulated surprise (p.148)."),
                "severity": "MEDIUM",
            })

    # Fear expression missing the hard-to-fake inner brow raise
    if emotions["fear"] > 0.35 and (elevated("eye_aperture") or elevated("mouth_open")):
        if absent("inner_brow_raise"):
            components.append(0.70)
            clues.append({
                "type": "MORPHOLOGY", "label": "Fear Without Fear Brow",
                "detail": ("Fear indicators active but AU1 inner-brow not elevated above "
                           "baseline — not a voluntary emblem; Ekman: its absence strongly "
                           "signals simulation (p.148)."),
                "severity": "HIGH",
            })

    # Anger brow draw without involuntary lower-eyelid tension
    if emotions["anger"] > 0.38 and depressed("brow_gap"):
        if absent("lower_lid_tension"):
            components.append(0.50)
            clues.append({
                "type": "MORPHOLOGY", "label": "Anger Without Eyelid Tension",
                "detail": ("Brow narrowed below baseline (AU4 — emblem, easily faked) but "
                           "AU7 lower-lid tension absent; Ekman: the one element missing in "
                           "simulated anger (p.149)."),
                "severity": "MEDIUM",
            })

    # Sadness lower face without the equally hard-to-fake sad brow
    if emotions["sadness"] > 0.38 and depressed("lip_corner_dir"):
        if absent("inner_brow_raise"):
            components.append(0.60)
            clues.append({
                "type": "MORPHOLOGY", "label": "Sadness Without Sad Brow",
                "detail": ("Lip corners below baseline but AU1 inner-brow not elevated — "
                           "Ekman: strongest clue that sadness is simulated (p.150)."),
                "severity": "HIGH",
            })

    # ── 2. TIMING ─────────────────────────────────────────────
    if len(history_features) >= 6:
        recent     = list(history_features)[-6:]
        onset_rate = abs(
            recent[-1].get("mouth_width", 0) - recent[0].get("mouth_width", 0)
        )
        if emotions["happiness"] > 0.38 and elevated("lip_corner_dir") and onset_rate > 0.06:
            components.append(0.45)
            clues.append({
                "type": "TIMING", "label": "Abrupt Expression Onset",
                "detail": ("Happiness reached full intensity in < 200 ms; Ekman: genuine "
                           "emotions build gradually — sharp onset signals posing."),
                "severity": "MEDIUM",
            })

        if emotions["surprise"] > 0.42 and baseline:
            bl_ob = baseline.get("outer_brow_raise", 0)
            bl_jd = baseline.get("jaw_drop", 0)
            n_sur = sum(
                1 for f in list(history_features)[-30:]
                if f.get("outer_brow_raise", 0) - bl_ob > 0.018
                and f.get("jaw_drop",         0) - bl_jd > 0.010
            )
            if n_sur > 20:
                components.append(0.70)
                clues.append({
                    "type": "TIMING", "label": "Prolonged Surprise",
                    "detail": (f"Surprise held ~{n_sur/30:.1f}s above baseline — "
                               "Ekman: genuine surprise is always brief (p.148)."),
                    "severity": "HIGH",
                })

    # ── 3. MICRO-EXPRESSIONS ──────────────────────────────────
    # Results passed in from process_frame (computed every 5 frames)
    # rather than re-running scipy here on every call.
    if micro_exprs is None:
        micro_exprs = detect_micro_expressions(history_features)
    for emotion_type, frame_idx in micro_exprs[-3:]:
        components.append(0.80)
        clues.append({
            "type": "MICRO-EXPRESSION",
            "label": f"Micro-Expression: {emotion_type}",
            "detail": (f"Brief suppressed {emotion_type.lower()} flash detected "
                       f"(1–5 frames, ~33–167 ms) at buffer position {frame_idx}."),
            "severity": "CRITICAL",
        })

    # ── 4. LEAKAGE ────────────────────────────────────────────
    if elevated("inner_brow_raise") and emotions["happiness"] > 0.38:
        components.append(0.60)
        clues.append({
            "type": "LEAKAGE", "label": "Fear/Sadness Brow With Happy Mouth",
            "detail": ("AU1 elevated above baseline (hard to suppress) while lower face "
                       "is managed into happiness — Ekman: brow reveals true felt emotion."),
            "severity": "HIGH",
        })

    if emotions["disgust"] > 0.38 and elevated("lower_lid_tension"):
        components.append(0.50)
        clues.append({
            "type": "LEAKAGE", "label": "Anger Leaking Through Disgust",
            "detail": ("Lower-lid tension above baseline behind disgust expression — "
                       "Ekman: anger may leak in the stare when disgust masks it (p.149)."),
            "severity": "MEDIUM",
        })

    # ── Score ─────────────────────────────────────────────────
    if components:
        base           = float(np.mean(components))
        quantity_boost = min(len(components) * 0.04, 0.12)
        raw_score      = min(base + quantity_boost, 0.92)
    else:
        raw_score = 0.04 + float(np.random.uniform(0, 0.04))

    smoothed_score = smoother.update(raw_score)
    return {
        "deception_score":    smoothed_score,
        "raw_score":          round(raw_score, 3),
        "clues":              clues,
        "dominant_emotion":   dominant,
        "emotion_confidence": round(dominant_conf, 3),
    }


# ══════════════════════════════════════════════════════════════
# Baseline calibration
# ══════════════════════════════════════════════════════════════

def compute_baseline(features_list: list) -> dict:
    """Compute per-feature mean from a list of feature dicts (neutral frames)."""
    if not features_list:
        return {}
    return {
        key: float(np.mean([f[key] for f in features_list if key in f]))
        for key in features_list[0]
    }


# ══════════════════════════════════════════════════════════════
# Main per-frame pipeline
# ══════════════════════════════════════════════════════════════

def _baseline_quality(frames: list) -> float:
    """1 - mean normalised std across features. High quality = stable neutral face."""
    if len(frames) < 5:
        return 0.5
    arr = np.array([[f[k] for k in FEATURE_NAMES] for f in frames])
    mean_std = float(np.mean(np.std(arr, axis=0)))
    return round(max(0.0, 1.0 - mean_std * 20), 2)


def _calibration_state(fc: int, history: list, baseline: dict) -> dict:
    """
    Return the calibration dict the frontend expects.
    Phases mirror the frontend updateCalibrationUI() logic:
      settling   — frames 1-4   (face stabilising, don't collect yet)
      collecting — frames 5-34  (actively building baseline, progress 0→100)
      active     — frames 35+   (baseline locked, full deception analysis)
    """
    if fc < 5:
        return {"phase": "settling",   "progress": 0,   "quality": 0.0}
    if fc < 35:
        progress = round((fc - 5) / 30 * 100)
        return {"phase": "collecting", "progress": progress, "quality": 0.0}
    quality = _baseline_quality(list(history)[:30]) if baseline else 0.5
    return {"phase": "active",     "progress": 100, "quality": quality}


def process_frame(img_bytes: str, state: dict) -> dict:
    """Decode one frame, run the full Ekman pipeline, return JSON-ready dict."""
    landmarker = get_face_landmarker()

    arr   = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result  = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return {"error": "no_face", "message": "No face detected"}

    lms      = result.face_landmarks[0]
    features = extract_features(lms, w, h)

    now = time.time()
    state["history"].append(features)
    state["timestamps"].append(now)
    state["frame_count"] += 1
    fc = state["frame_count"]

    if fc == 35:
        state["baseline"] = compute_baseline(list(state["history"])[:30])

    baseline = state["baseline"]

    # Delta features — computed once, reused by both emotion classifier and
    # deception analyser so we don't iterate the dict twice.
    if baseline:
        adj = {k: features[k] - baseline.get(k, features[k]) for k in features}
    else:
        adj = {k: 0.0 for k in features}

    # ── FIX 1: CNN runs in a background thread — never blocks this response ──
    # Fire a new CNN inference only when the previous one has finished.
    # The most recent cached result is used immediately for this frame.
    CNN_EVERY = 3   # tightened from 5 — CNN is now off-thread so cost is ~0
    if fc % CNN_EVERY == 0 and not state["cnn_running"]:
        # Build face crop — single pass over landmarks (FIX 4)
        xs = [p.x * w for p in lms]
        ys = [p.y * h for p in lms]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad_x = int(0.15 * (xmax - xmin))
        pad_y = int(0.15 * (ymax - ymin))
        x1 = int(max(0, xmin - pad_x))
        y1 = int(max(0, ymin - pad_y))
        x2 = int(min(w, xmax + pad_x))
        y2 = int(min(h, ymax + pad_y))
        face_crop = frame[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else frame

        state["cnn_running"] = True
        t = threading.Thread(
            target=_run_cnn_async,
            args=(state, face_crop),
            daemon=True,
        )
        t.start()

    # Use whatever CNN result is cached — updated async in the background
    emotions = state["last_emotions"]

    # ── FIX 3: Convert deque to list once — reused by micro + deception ──
    history_list = list(state["history"])

    # ── FIX 2: Micro-expressions only every 5 frames (scipy is non-trivial) ──
    # detect_micro_expressions already clamps to last 30 frames internally.
    if fc % 5 == 0:
        state["last_micro"] = detect_micro_expressions(history_list)
    micro_exprs = state.get("last_micro", [])

    deception = analyse_deception(
        features,
        baseline or {},
        emotions,
        history_list,
        list(state["timestamps"]),
        state["smoother"],
        micro_exprs,
    )

    def pts(indices):
        return [(int(lms[i].x * w), int(lms[i].y * h))
                for i in indices if i < len(lms)]

    regions = {
        "brow_left":  pts(LANDMARKS["inner_brow_left"]  + LANDMARKS["outer_brow_left"]),
        "brow_right": pts(LANDMARKS["inner_brow_right"] + LANDMARKS["outer_brow_right"]),
        "eye_left":   pts(LANDMARKS["upper_lid_left"]   + LANDMARKS["lower_lid_left"]),
        "eye_right":  pts(LANDMARKS["upper_lid_right"]  + LANDMARKS["lower_lid_right"]),
        "mouth":      pts(LANDMARKS["mouth_corner_left"] + LANDMARKS["mouth_corner_right"] +
                          LANDMARKS["upper_lip"][:5] + LANDMARKS["lower_lip"][:5]),
        "nose":       pts(LANDMARKS["nose_bridge"] + LANDMARKS["nose_lower"]),
    }

    cal = _calibration_state(fc, state["history"], state["baseline"])
    return {
        "status":         "ok",
        "frame":          fc,
        "elapsed":        round(now - state["start_time"], 1),
        "features":       features,
        "emotions":       emotions,
        "deception":      deception,
        "regions":        regions,
        "calibration":    cal,
        "calibrating":    fc < 35,
        "baseline_ready": bool(state["baseline"]),
    }



# ══════════════════════════════════════════════════════════════
# Flask routes
# ══════════════════════════════════════════════════════════════



@app.route("/train")
def train_page():
    """Serve the in-browser expression training tool."""
    return render_template("train.html")

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    try:
        return render_template("index.html")
    except Exception as e:
        # Template missing — return a diagnostic page so the error is visible
        return (
            f"<h2>VERITY — Template Error</h2>"
            f"<p><b>{type(e).__name__}:</b> {e}</p>"
            f"<p>Make sure <code>templates/index.html</code> exists in your repo.</p>"
            f"<p>Backend status: <a href='/health'>/health</a> | "
            f"<a href='/warmup'>/warmup</a></p>"
        ), 500


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Prefer the session_id sent by the frontend JS (crypto.randomUUID).
    # This is more reliable than Flask cookie sessions on cloud platforms
    # where cookie domain / SameSite settings can silently break state.
    sid = (data.get("session_id") or "").strip()
    if not sid:
        # Fallback: Flask cookie session
        if "sid" not in session:
            session["sid"] = str(uuid.uuid4())
        sid = session["sid"]

    state = get_session_state(sid)

    img_b64 = data["image"].split(",")[-1]
    try:
        result = process_frame(img_b64, state)
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"error": "processing_error", "message": str(exc)}), 500
    return jsonify(result)


@app.route("/reset", methods=["POST"])
def reset():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]
    with _session_lock:
        _session_store.pop(sid, None)   # next request re-creates fresh state
    return jsonify({"status": "reset"})


@app.route("/health")
def health():
    # This route MUST respond in milliseconds — Render uses it to decide
    # if the dyno is alive.  Never wait on model/classifier here.
    with _session_lock:
        n_sessions = len(_session_store)
    model_status  = "ready" if os.path.exists(_MODEL_PATH) else "not downloaded"
    clf_status    = "ready" if _emotion_clf is not None else "pending (lazy)"
    return jsonify({
        "status":     "ok",
        "model":      model_status,
        "classifier": clf_status,
        "sessions":   n_sessions,
    })


# ══════════════════════════════════════════════════════════════
# Video analysis route  (unchanged API; improved internals)
# ══════════════════════════════════════════════════════════════

@app.route("/analyse_video", methods=["POST"])
def analyse_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = ".mp4" if "mp4" in video_file.content_type else (
             ".ogv"  if "ogg" in video_file.content_type else ".webm")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    try:
        result = _process_video_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return jsonify(result)


def _process_video_file(path: str) -> dict:
    """Frame-by-frame video analysis with per-file isolated state."""
    # Dedicated landmarker instance for this video (not shared with live stream)
    vfm_opts = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    vfm = _mp_vision.FaceLandmarker.create_from_options(vfm_opts)
    # Each video file gets its own smoother — no cross-contamination
    v_smoother = ExponentialSmoother(alpha=0.15)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps_src
    SAMPLE_EVERY = 3

    timeline      = []
    all_features  = []
    all_times     = []
    v_baseline    = {}
    v_frame       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        v_frame += 1
        if v_frame % SAMPLE_EVERY != 0:
            continue

        ts   = v_frame / fps_src
        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res     = vfm.detect(mp_img)

        if not res.face_landmarks:
            timeline.append({"t": round(ts, 2), "frame": v_frame, "face": False})
            continue

        lms = res.face_landmarks[0]
        features = extract_features(lms, w, h)
        v_adj    = ({k: features[k] - v_baseline.get(k, features[k]) for k in features}
                   if v_baseline else {k: 0.0 for k in features})
        emotions = classify_emotion(v_adj)

        all_features.append(features)
        all_times.append(ts)

        if len(all_features) == 30:
            v_baseline = compute_baseline(all_features[:30])

        adj = features.copy()
        if v_baseline:
            for k in adj:
                if k in v_baseline:
                    adj[k] = features[k] - v_baseline[k]

        deception = analyse_deception(
            features,
            v_baseline or {},
            emotions,
            all_features,
            all_times,
            v_smoother,
        )

        thumb_b64 = None
        if deception["deception_score"] > 0.55 or deception["clues"]:
            small = cv2.resize(frame, (120, 90))
            _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

        timeline.append({
            "t":               round(ts, 2),
            "frame":           v_frame,
            "face":            True,
            "emotions":        emotions,
            "features":        {k: round(v, 3) for k, v in features.items()},
            "deception_score": deception["deception_score"],
            "raw_score":       deception["raw_score"],
            "clues":           deception["clues"],
            "dominant":        deception["dominant_emotion"],
            "thumb":           thumb_b64,
        })

    cap.release()
    vfm.close()

    if not timeline:
        return {"error": "No frames processed"}

    face_samples = [s for s in timeline if s.get("face")]
    scores       = [s["deception_score"] for s in face_samples]

    all_clues = [
        {**c, "t": s["t"]}
        for s in face_samples
        for c in s.get("clues", [])
    ]

    key_moments = sorted(
        [s for s in face_samples if s.get("clues") and s.get("thumb")],
        key=lambda x: x["deception_score"],
        reverse=True,
    )[:5]

    clue_freq: dict = {}
    for c in all_clues:
        clue_freq[c["label"]] = clue_freq.get(c["label"], 0) + 1

    summary = {
        "duration":          round(duration, 1),
        "total_frames":      total_frames,
        "samples_analysed":  len(face_samples),
        "face_coverage":     round(len(face_samples) / max(len(timeline), 1) * 100, 1),
        "avg_deception":     round(float(np.mean(scores)) * 100, 1) if scores else 0,
        "max_deception":     round(float(np.max(scores))  * 100, 1) if scores else 0,
        "verdict":           _verdict(float(np.mean(scores)) if scores else 0),
        "clue_freq":         dict(sorted(clue_freq.items(), key=lambda x: -x[1])),
        "key_moments":       key_moments,
    }

    for s in timeline:
        s.pop("thumb", None)   # thumbnails only in key_moments

    return {"status": "ok", "summary": summary, "timeline": timeline}


def _verdict(score: float) -> str:
    if score < 0.25:   return "TRUTHFUL"
    if score < 0.55:   return "INCONCLUSIVE"
    return "DECEPTION LIKELY"


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  VERITY v2 — DECEPTION DETECTION ENGINE")
    print("  Ekman/Friesen · sklearn MLP · scipy peaks · EMA smoother")
    print("  http://localhost:5050")
    print("═" * 60 + "\n")
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, host="0.0.0.0", threaded=True)




@app.route("/ai_analysis", methods=["POST"])
def ai_analysis():
    """
    Claude-powered forensic narrative from live session history.
    Requires ANTHROPIC_API_KEY environment variable.
    Frontend sends: {session_id, history: [{deception_score, clues, emotions, elapsed}]}
    """
    data = request.get_json() or {}
    history = data.get("history", [])
    if not history:
        return jsonify({"status": "error", "message": "No session history provided"}), 400

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({
            "status": "error",
            "message": "ANTHROPIC_API_KEY not set. Add it in Render → Environment."
        }), 503

    # Summarise history for the prompt
    n        = len(history)
    avg_dec  = round(sum(s.get("deception_score", 0) for s in history) / max(n, 1) * 100, 1)
    max_dec  = round(max((s.get("deception_score", 0) for s in history), default=0) * 100, 1)
    all_clues = [c["label"] for s in history for c in s.get("clues", [])]
    top_clues = sorted(set(all_clues), key=lambda x: -all_clues.count(x))[:5]

    emotion_totals = {}
    for s in history:
        for emo, val in s.get("emotions", {}).items():
            emotion_totals[emo] = emotion_totals.get(emo, 0) + val
    dominant_emo = max(emotion_totals, key=emotion_totals.get) if emotion_totals else "unknown"

    prompt = f"""You are VERITY, a forensic facial expression analyst trained in Paul Ekman's 
Unmasking the Face (2003) framework. Provide a concise forensic narrative (150-200 words) 
of the following live analysis session.

Session data ({n} frames, ~{round(n*0.12)}s):
- Average deception index: {avg_dec}%
- Peak deception index: {max_dec}%
- Dominant emotion: {dominant_emo}
- Top deception indicators detected: {', '.join(top_clues) if top_clues else 'none'}

Write in the style of a professional forensic report. Reference specific Ekman principles 
where relevant. Be precise about what the indicators suggest. Do not speculate beyond the data."""

    try:
        import urllib.request as _req, json as _json
        payload = _json.dumps({
            "model":      "claude-sonnet-4-20250514",
            "max_tokens": 400,
            "messages":   [{"role": "user", "content": prompt}],
        }).encode()
        req = _req.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with _req.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
        text = result["content"][0]["text"]
        return jsonify({"status": "ok", "analysis": text})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/debug")
def debug():
    """
    Diagnostic endpoint — visit this URL to see exactly what is happening.
    Share the JSON output if reporting an issue.
    """
    import sys, platform

    model_exists = os.path.exists(_MODEL_PATH)
    model_size   = round(os.path.getsize(_MODEL_PATH) / 1e6, 1) if model_exists else 0

    # Try a quick face-detection smoke test (no image — just creates the landmarker)
    landmarker_ok  = False
    landmarker_err = None
    try:
        get_face_landmarker()
        landmarker_ok = True
    except Exception as e:
        landmarker_err = str(e)

    clf_ok  = False
    clf_err = None
    try:
        _get_classifier()
        clf_ok = True
    except Exception as e:
        clf_err = str(e)

    with _session_lock:
        n_sessions = len(_session_store)

    return jsonify({
        "python":          sys.version,
        "platform":        platform.platform(),
        "model_path":      _MODEL_PATH,
        "model_exists":    model_exists,
        "model_size_mb":   model_size,
        "landmarker_ok":   landmarker_ok,
        "landmarker_err":  landmarker_err,
        "classifier_ok":   clf_ok,
        "classifier_err":  clf_err,
        "sessions":        n_sessions,
        "startup_log":     _STARTUP_LOG[-20:],
        "startup_errors":  _STARTUP_ERRORS,
        "mp_version":      getattr(__import__("mediapipe"), "__version__", "unknown"),
    })


# ══════════════════════════════════════════════════════════════
# TRAINING SYSTEM
# Collects labelled feature vectors from real expressions and
# trains an sklearn MLP that replaces the rule-based classifier.
#
# Flow:
#   1. Browser: POST /collect_sample {features, label} × N frames
#   2. Browser: POST /train_model
#   3. Backend trains MLP, saves to face_model.pkl
#   4. classify_emotion() automatically uses the trained model
# ══════════════════════════════════════════════════════════════
import joblib

_TRAINING_STORE: list[dict] = []   # [{features: {...}, label: str}]
_TRAINED_MODEL  = None             # sklearn Pipeline once trained
_TRAINED_SCALER = None
_MODEL_PKL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_model.pkl"
)

# Load saved model on startup if it exists
try:
    if os.path.exists(_MODEL_PKL_PATH):
        _TRAINED_MODEL = joblib.load(_MODEL_PKL_PATH)
        print(f"[VERITY] Loaded trained emotion model from {_MODEL_PKL_PATH}", flush=True)
except Exception as _e:
    print(f"[VERITY] Could not load trained model: {_e}", flush=True)


@app.route("/collect_sample", methods=["POST"])
def collect_sample():
    """Store one labelled feature vector for training."""
    data = request.get_json() or {}
    features = data.get("features")
    label    = data.get("label")
    if not features or not label:
        return jsonify({"error": "features and label required"}), 400
    if label not in EMOTION_LABELS:
        return jsonify({"error": f"label must be one of {EMOTION_LABELS}"}), 400
    if not all(k in features for k in FEATURE_NAMES):
        return jsonify({"error": "features dict missing keys"}), 400
    _TRAINING_STORE.append({"features": features, "label": label})
    counts = {em: sum(1 for s in _TRAINING_STORE if s["label"] == em) for em in EMOTION_LABELS}
    return jsonify({"status": "ok", "total_samples": len(_TRAINING_STORE), "counts": counts})


@app.route("/reset_training", methods=["POST"])
def reset_training():
    """Clear all collected training samples."""
    _TRAINING_STORE.clear()
    return jsonify({"status": "ok", "total_samples": 0})


@app.route("/training_status")
def training_status():
    """Return counts of collected samples per emotion."""
    counts = {em: sum(1 for s in _TRAINING_STORE if s["label"] == em) for em in EMOTION_LABELS}
    model_ready = _TRAINED_MODEL is not None
    return jsonify({
        "total_samples": len(_TRAINING_STORE),
        "counts":        counts,
        "model_ready":   model_ready,
        "model_path":    _MODEL_PKL_PATH if model_ready else None,
    })


@app.route("/train_model", methods=["POST"])
def train_model():
    """
    Train an sklearn MLP on collected samples and save to disk.
    Requires at least 10 samples per emotion (60 total).
    """
    global _TRAINED_MODEL

    if len(_TRAINING_STORE) < 60:
        return jsonify({
            "error": f"Need at least 60 samples (10 per emotion), have {len(_TRAINING_STORE)}"
        }), 400

    counts = {em: sum(1 for s in _TRAINING_STORE if s["label"] == em) for em in EMOTION_LABELS}
    sparse = [em for em, n in counts.items() if n < 5]
    if sparse:
        return jsonify({"error": f"Too few samples for: {sparse}. Need ≥5 per emotion."}), 400

    X = np.array([[s["features"][k] for k in FEATURE_NAMES] for s in _TRAINING_STORE])
    y = np.array([EMOTION_LABELS.index(s["label"]) for s in _TRAINING_STORE])

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )),
    ])
    clf.fit(X, y)

    # Score on training data
    train_acc = float(clf.score(X, y))

    joblib.dump(clf, _MODEL_PKL_PATH)
    _TRAINED_MODEL = clf

    print(f"[VERITY] Trained model saved — accuracy {train_acc:.1%} on {len(X)} samples", flush=True)
    return jsonify({
        "status":     "ok",
        "accuracy":   round(train_acc, 3),
        "n_samples":  len(X),
        "counts":     counts,
        "model_path": _MODEL_PKL_PATH,
    })

@app.route("/warmup")
def warmup():
    """
    Trigger model and classifier initialisation synchronously.
    The frontend can GET /warmup once at page load and wait for the
    200 response before starting the analysis loop. This guarantees
    the first /analyse frame returns instantly.
    """
    try:
        _ensure_model()
        _get_classifier()   # also warms up sklearn pipeline
        model_ok = os.path.exists(_MODEL_PATH)
        clf_ok   = _emotion_clf is not None
        return jsonify({
            "status":     "ready" if (model_ok and clf_ok) else "partial",
            "model":      "ok" if model_ok else "missing",
            "classifier": "ok" if clf_ok else "missing",
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
