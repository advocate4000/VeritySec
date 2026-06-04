"""
Deception Detection Engine
Based on Paul Ekman & Wallace V. Friesen's "Unmasking the Face" (2003)
Uses MediaPipe Face Mesh + Ekman's morphology, timing & micro-expression framework
"""

import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import time
import math
import os
import tempfile
import urllib.request
from collections import deque
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

# ──────────────────────────────────────────────────────────
# MediaPipe setup — Tasks API (mp.solutions removed in ≥0.10.8)
# ──────────────────────────────────────────────────────────
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

def _ensure_model():
    """Download FaceLandmarker model on first run (~5 MB). Re-downloads if corrupted."""
    MIN_SIZE = 1_000_000  # real model is ~5 MB; anything under 1 MB is corrupt
    corrupt = os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) < MIN_SIZE
    if corrupt:
        print("[VERITY] Model file appears corrupt — re-downloading...", flush=True)
        os.remove(_MODEL_PATH)
    if not os.path.exists(_MODEL_PATH):
        print("[VERITY] Downloading FaceLandmarker model (~5 MB) …", flush=True)
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("[VERITY] Model saved to", _MODEL_PATH, flush=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download FaceLandmarker model: {exc}\n"
                f"Download manually from:\n  {_MODEL_URL}\n"
                f"and place it at: {_MODEL_PATH}"
            ) from exc

_ensure_model()

def _make_landmarker(detection_conf=0.5, tracking_conf=0.5):
    """Create a FaceLandmarker in IMAGE mode (stateless, safe for per-frame use)."""
    opts = _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=_mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=detection_conf,
        min_face_presence_confidence=tracking_conf,
        min_tracking_confidence=tracking_conf,
    )
    return _mp_vision.FaceLandmarker.create_from_options(opts)

# Lazy init — do NOT create the landmarker at import time.
# Gunicorn --preload forks after import; TFLite (used by MediaPipe) is not
# fork-safe, so a module-level instance causes "post_fork hook error" crashes.
# Each worker creates its own instance on its first request instead.
_face_landmarker = None

def get_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        _face_landmarker = _make_landmarker()
    return _face_landmarker

# ──────────────────────────────────────────────────────────
# Ekman-derived landmark regions (MediaPipe 468-point mesh)
# ──────────────────────────────────────────────────────────
# Source: Ekman Ch.11 "Facial Deceit" — morphology analysis focuses on:
#   Lower face (mouth/lips): primary management zone — most controlled
#   Upper face (brow/forehead): primary LEAKAGE zone — hardest to fake

LANDMARKS = {
    # Inner brow — Ekman: fear & sad brow hardest to voluntarily produce; reliable leakage site
    "inner_brow_left":  [107, 66, 105, 63, 70],
    "inner_brow_right": [336, 296, 334, 293, 300],
    # Outer brow — anger/concentration emblem; more voluntarily controlled
    "outer_brow_left":  [46, 53, 52, 65, 55],
    "outer_brow_right": [285, 295, 282, 283, 276],
    # Upper eyelid — tension indicator for anger; pulled up inner corner in sadness
    "upper_lid_left":   [159, 158, 157, 173, 133],
    "upper_lid_right":  [386, 385, 384, 398, 362],
    # Lower eyelid — Duchenne smile marker; tension in anger
    "lower_lid_left":   [145, 144, 163, 7],
    "lower_lid_right":  [374, 373, 390, 249],
    # Cheeks — raised in genuine happiness (Duchenne)
    "cheek_left":       [116, 117, 118, 119, 120],
    "cheek_right":      [345, 346, 347, 348, 349],
    # Mouth corners — Ekman: primary management site; lip corner pull = happiness
    "mouth_corner_left":  [61, 146, 91, 181, 84],
    "mouth_corner_right": [291, 375, 321, 405, 314],
    # Upper lip — disgust upper lip raise; fear lip stretching
    "upper_lip":        [13, 312, 311, 310, 415, 308, 78, 191, 80, 81, 82],
    # Lower lip — sadness; quivering chin below
    "lower_lip":        [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13],
    # Jaw / chin — fear lip stretch pulls here
    "jaw":              [152, 377, 400, 378, 379, 365, 397, 288],
    # Nose bridge — disgust nose wrinkle
    "nose_bridge":      [6, 197, 195, 5],
    "nose_lower":       [4, 240, 98, 97, 2, 326, 327, 460],
}

# ──────────────────────────────────────────────────────────
# Temporal state (per-session deque buffers)
# ──────────────────────────────────────────────────────────
HISTORY_LEN = 90  # ~3 seconds at 30fps
expression_history = deque(maxlen=HISTORY_LEN)
timestamp_history  = deque(maxlen=HISTORY_LEN)
baseline_features  = {}
baseline_sigma     = {}   # per-feature std dev during neutral — personalised noise floors
feature_peaks      = {}   # {expr: {feature: raw_mean}} — personalised range scaling
frame_count        = 0
session_start      = time.time()

# ── Voice state ───────────────────────────────────────────
VOICE_HISTORY_LEN = 90          # ~11s at one sample per 120ms (extended for pause analysis)
voice_history  = deque(maxlen=VOICE_HISTORY_LEN)
voice_baseline = {}             # mean pitch / rms + new: jitter, syllable_rate, pause_rate

# ── rPPG (remote photoplethysmography) ───────────────────
# Extracts heart rate from subtle skin colour changes in the forehead ROI.
# Green channel carries the strongest blood-volume pulse; normalized against
# total intensity to cancel ambient light drift.
RPPG_BUFFER_LEN = 250          # ~31s at 8fps (120ms send interval)
rppg_buffer  = deque(maxlen=RPPG_BUFFER_LEN)   # {t, g_norm} per frame
hr_history   = deque(maxlen=60)                  # rolling BPM estimates
hr_baseline  = 0.0             # resting BPM captured at SET BASELINE
hr_signal_quality = 0.0        # 0-1, updated each computation
question_current    = 0         # 0 = no questions marked yet
question_markers    = []        # [{question, elapsed, frame}]
question_voice_data = {}        # {q_num: {scores:[], clues:[], pitches:[], jitters:[], syllable_rates:[]}}

# ── EMA smoothing (α=0.35: fast enough to feel live, slow enough to cut noise) ──
# Applied server-side so the frontend renders stable values without any extra logic.
EMA_ALPHA   = 0.35
emotion_ema: dict = {}

# ──────────────────────────────────────────────────────────
# Helper geometry
# ──────────────────────────────────────────────────────────

def lm(landmarks, idx, w, h):
    """Return (x, y) pixel coordinate for landmark index."""
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h])


def region_centroid(landmarks, indices, w, h):
    pts = np.array([lm(landmarks, i, w, h) for i in indices])
    return pts.mean(axis=0)


def region_spread(landmarks, indices, w, h):
    """Mean distance of region points from their centroid — proxy for muscle tension."""
    pts = np.array([lm(landmarks, i, w, h) for i in indices])
    c = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))


def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def angle_of_lift(landmarks, tip_idx, base_idx, w, h):
    """Vertical lift of a point relative to a base point (positive = upward)."""
    tip  = lm(landmarks, tip_idx,  w, h)
    base = lm(landmarks, base_idx, w, h)
    return float(base[1] - tip[1])  # image Y inverted

# ──────────────────────────────────────────────────────────
# Feature extraction (all normalised to face-width)
# ──────────────────────────────────────────────────────────

def extract_features(landmarks, w, h):
    """
    Extract scalar muscle-activity features per Ekman facial zones.
    All distances normalised by inter-ocular distance for scale invariance.
    """
    # Normalisation baseline: inter-ocular distance
    left_eye_outer  = lm(landmarks, 33,  w, h)
    right_eye_outer = lm(landmarks, 263, w, h)
    iod = max(euclidean(left_eye_outer, right_eye_outer), 1.0)

    def norm(d): return d / iod

    # ── Brow / Forehead ───────────────────────────────────
    # Inner brow height — fear/sadness brow raise (hardest to fake; Ekman p.148–150)
    inner_brow_L = angle_of_lift(landmarks, 107, 33,  w, h)
    inner_brow_R = angle_of_lift(landmarks, 336, 263, w, h)
    inner_brow_raise = norm((inner_brow_L + inner_brow_R) / 2)

    # Brow draw-together — anger/concentration (also an emblem; p.149)
    brow_gap = norm(euclidean(lm(landmarks, 107, w, h), lm(landmarks, 336, w, h)))

    # Brow outer raise — surprise brow (emblem; easy to fake)
    outer_brow_L = angle_of_lift(landmarks, 70,  33,  w, h)
    outer_brow_R = angle_of_lift(landmarks, 300, 263, w, h)
    outer_brow_raise = norm((outer_brow_L + outer_brow_R) / 2)

    # ── Eyes ──────────────────────────────────────────────
    # Eye aperture — fear stare (wide) vs relaxed
    eye_top_L    = lm(landmarks, 159, w, h)
    eye_bot_L    = lm(landmarks, 145, w, h)
    eye_top_R    = lm(landmarks, 386, w, h)
    eye_bot_R    = lm(landmarks, 374, w, h)
    eye_open_L   = norm(euclidean(eye_top_L, eye_bot_L))
    eye_open_R   = norm(euclidean(eye_top_R, eye_bot_R))
    eye_aperture = (eye_open_L + eye_open_R) / 2

    # Lower lid tension — Duchenne happiness & anger; often absent in fake expressions
    lower_lid_L = region_spread(landmarks, LANDMARKS["lower_lid_left"],  w, h)
    lower_lid_R = region_spread(landmarks, LANDMARKS["lower_lid_right"], w, h)
    lower_lid_tension = norm((lower_lid_L + lower_lid_R) / 2)

    # Cheek raise — Duchenne marker (Ekman: absent in simulated happiness)
    cheek_L = angle_of_lift(landmarks, 116, 145, w, h)
    cheek_R = angle_of_lift(landmarks, 345, 374, w, h)
    cheek_raise = norm((cheek_L + cheek_R) / 2)

    # ── Mouth / Lower face (primary management zone; Ekman p.146) ────
    # Mouth width — happiness corner pull; fear lip stretch
    mouth_L = lm(landmarks, 61,  w, h)
    mouth_R = lm(landmarks, 291, w, h)
    mouth_width = norm(euclidean(mouth_L, mouth_R))

    # Mouth height — open/closed; fear jaw drop; surprise
    mouth_top = lm(landmarks, 13,  w, h)
    mouth_bot = lm(landmarks, 14,  w, h)
    mouth_open = norm(euclidean(mouth_top, mouth_bot))

    # Lip corner direction — up (happiness) vs down (sadness/disgust)
    nose_tip = lm(landmarks, 4, w, h)
    corner_L_up = norm(float(nose_tip[1] - mouth_L[1]))
    corner_R_up = norm(float(nose_tip[1] - mouth_R[1]))
    lip_corner_dir = (corner_L_up + corner_R_up) / 2

    # Upper lip raise — disgust (nose-to-lip distance shrinks)
    nose_base   = lm(landmarks, 2,  w, h)
    upper_lip_c = lm(landmarks, 13, w, h)
    upper_lip_raise = norm(float(nose_base[1] - upper_lip_c[1]))

    # Lip press — anger lip compression (lip height minimal, corners level)
    lip_press = 1.0 - min(mouth_open * 10, 1.0)

    # Nose wrinkle — disgust (spread of nose-bridge landmarks)
    nose_wrinkle = norm(region_spread(landmarks, LANDMARKS["nose_lower"], w, h))

    # Jaw drop — surprise / fear
    jaw_center  = lm(landmarks, 152, w, h)
    jaw_drop = norm(float(jaw_center[1] - mouth_bot[1]))

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

# ──────────────────────────────────────────────────────────
# Emotion classifier (rule-based, Ekman action units)
# ──────────────────────────────────────────────────────────

def classify_emotion(f, is_delta=False):
    """
    Classify primary emotion using Ekman action units.

    is_delta=False  — raw features, pre-calibration. Fixed floors.
    is_delta=True   — baseline-subtracted deltas. Uses per-person sigma as
                      noise floor (2.5σ) and per-person expression peaks for
                      range normalisation once guided calibration is complete.

    Normalisation principle (delta mode):
      score = gate(delta, 2.5σ) / peak_range
    where peak_range is the observed maximum delta during the calibration
    expression for that feature. Score of 1.0 = the user's personal maximum.
    """
    scores = {}

    if is_delta:
        def g(val, fk):
            noise = max(2.5 * baseline_sigma.get(fk, 0), 0.003) if baseline_sigma else 0.003
            return max(val - noise, 0.0)

        def n(gated, fk, ek, fm=6.0):
            if not gated:
                return 0.0
            if feature_peaks and ek in feature_peaks and baseline_features:
                raw_pk = feature_peaks[ek].get(fk)
                if raw_pk is not None:
                    peak_d = abs(raw_pk - baseline_features.get(fk, 0))
                    noise  = max(2.5 * baseline_sigma.get(fk, 0), 0.003) if baseline_sigma else 0.003
                    denom  = max(peak_d - noise, 0.005)
                    return min(gated / denom, 1.0)
            return min(gated * fm, 1.0)

        brow_noise = max(2.5 * baseline_sigma.get('brow_gap', 0), 0.003) if baseline_sigma else 0.003
        brow_raw   = max(-f["brow_gap"] - brow_noise, 0)
        if feature_peaks and 'anger' in feature_peaks and baseline_features:
            pk_gap     = feature_peaks['anger'].get('brow_gap', baseline_features.get('brow_gap', 0.35))
            bl_gap     = baseline_features.get('brow_gap', 0.35)
            brow_range = max(bl_gap - pk_gap - brow_noise, 0.005)
            brow_draw  = min(brow_raw / brow_range, 1.0)
        else:
            brow_draw = min(brow_raw * 5.0, 1.0)

        scores["happiness"] = np.clip(
            0.40 * n(g(f["lip_corner_dir"],   "lip_corner_dir"),   "lip_corner_dir",   "smile") +
            0.35 * n(g(f["cheek_raise"],       "cheek_raise"),      "cheek_raise",      "smile") +
            0.25 * n(g(f["lower_lid_tension"], "lower_lid_tension"),"lower_lid_tension","smile"),
            0, 1)

        scores["surprise"] = np.clip(
            0.40 * n(g(f["outer_brow_raise"], "outer_brow_raise"), "outer_brow_raise", "eyebrows") +
            0.25 * n(g(f["eye_aperture"],     "eye_aperture"),     "eye_aperture",     "eyebrows") +
            0.35 * n(g(f["jaw_drop"],         "jaw_drop"),         "jaw_drop",         "eyebrows"),
            0, 1)

        scores["fear"] = np.clip(
            0.50 * n(g(f["inner_brow_raise"], "inner_brow_raise"), "inner_brow_raise", "eyebrows") +
            0.20 * n(g(f["eye_aperture"],     "eye_aperture"),     "eye_aperture",     "eyebrows") +
            0.20 * n(g(f["mouth_open"],       "mouth_open"),       "mouth_open",       "eyebrows") +
            0.10 * n(g(f["mouth_width"],      "mouth_width"),      "mouth_width",      "eyebrows"),
            0, 1)

        scores["anger"] = np.clip(
            0.40 * brow_draw +
            0.35 * n(g(f["lower_lid_tension"], "lower_lid_tension"), "lower_lid_tension", "anger") +
            0.15 * n(g(f["lip_press"],         "lip_press"),         "lip_press",         "anger") +
            0.10 * n(g(f["eye_aperture"],      "eye_aperture"),      "eye_aperture",      "anger"),
            0, 1)

        scores["disgust"] = np.clip(
            0.45 * n(g(f["upper_lip_raise"], "upper_lip_raise"), "upper_lip_raise", "anger") * 1.5 +
            0.35 * n(g(f["nose_wrinkle"],    "nose_wrinkle"),    "nose_wrinkle",    "anger") +
            0.20 * n(g(-f["lip_corner_dir"], "lip_corner_dir"),  "lip_corner_dir",  "anger"),
            0, 1)

        scores["sadness"] = np.clip(
            0.45 * n(g(f["inner_brow_raise"],  "inner_brow_raise"), "inner_brow_raise", "anger") +
            0.35 * n(g(-f["lip_corner_dir"],   "lip_corner_dir"),   "lip_corner_dir",   "anger") +
            0.20 * n(g(-f["cheek_raise"],      "cheek_raise"),      "cheek_raise",      "anger"),
            0, 1)

    else:
        def gate(val, floor):
            return max(val - floor, 0.0)

        brow_draw = max(0.33 - f["brow_gap"], 0) * 2.0

        scores["happiness"] = np.clip(
            0.40 * gate(f["lip_corner_dir"],    0.015) +
            0.35 * gate(f["cheek_raise"],       0.010) +
            0.25 * gate(f["lower_lid_tension"], 0.130),
            0, 1)

        scores["surprise"] = np.clip(
            0.40 * gate(f["outer_brow_raise"], 0.010) +
            0.25 * gate(f["eye_aperture"],     0.165) +
            0.35 * gate(f["jaw_drop"],         0.005),
            0, 1)

        scores["fear"] = np.clip(
            0.50 * gate(f["inner_brow_raise"], 0.008) +
            0.20 * gate(f["eye_aperture"],     0.165) +
            0.20 * gate(f["mouth_open"],       0.020) +
            0.10 * gate(f["mouth_width"],      0.300),
            0, 1)

        scores["anger"] = np.clip(
            0.40 * brow_draw +
            0.35 * gate(f["lower_lid_tension"], 0.150) +
            0.15 * gate(f["lip_press"],         0.720) +
            0.10 * gate(f["eye_aperture"],      0.165),
            0, 1)

        scores["disgust"] = np.clip(
            0.45 * gate(f["upper_lip_raise"], 0.060) * 2.5 +
            0.35 * gate(f["nose_wrinkle"],    0.090) +
            0.20 * gate(-f["lip_corner_dir"], 0.010),
            0, 1)

        scores["sadness"] = np.clip(
            0.45 * gate(f["inner_brow_raise"],  0.008) +
            0.35 * gate(-f["lip_corner_dir"],   0.010) +
            0.20 * gate(-f["cheek_raise"],      0.010),
            0, 1)

    total = sum(scores.values()) + 1e-6
    return {k: round(float(v / total), 3) for k, v in scores.items()}



# ──────────────────────────────────────────────────────────
# Ekman Deception Analysis
# Implements: morphology, timing, micro-expression detection
# Source: "Unmasking the Face" Ch.11 "Facial Deceit" pp.143-155
# ──────────────────────────────────────────────────────────

def analyse_deception(features, emotions, history_features, history_times):
    """
    Apply Ekman's four deception-detection criteria:
    1. Morphology  — upper vs lower face incongruence
    2. Timing      — onset, duration, offset anomalies
    3. Micro-exprs — brief suppressed flashes
    4. Leakage     — felt emotion bleeding through managed expression
    Returns deception_score (0-1) + explanations
    """
    clues = []
    deception_components = []

    dominant = max(emotions, key=emotions.get)
    dominant_conf = emotions[dominant]

    # ── 1. MORPHOLOGY: Upper-lower face incongruence ──────────────
    # Ekman p.146: "more effort is focused on managing the mouth...
    #  brow/forehead area leaks genuine felt emotion"

    # Fake happiness: smile present but Duchenne markers absent
    if emotions["happiness"] > 0.35:
        if features["cheek_raise"] < 0.02 or features["lower_lid_tension"] < 0.15:
            deception_components.append(0.65)
            clues.append({
                "type": "MORPHOLOGY",
                "label": "Non-Duchenne Smile",
                "detail": "Lip corners raised (mouth region managed) but cheek & lower-lid muscle engagement absent — hallmark of a social smile per Ekman",
                "severity": "HIGH"
            })

    # Fake surprise: missing relaxed eyelid opening (Ekman p.148)
    if emotions["surprise"] > 0.30:
        if features["eye_aperture"] < 0.18:
            deception_components.append(0.55)
            clues.append({
                "type": "MORPHOLOGY",
                "label": "Suppressed Eyelid Opening",
                "detail": "Surprise brow raised (emblem — easy to produce) but relaxed eyelid opening absent; Ekman: only deception clue in simulated surprise",
                "severity": "MEDIUM"
            })

    # Fear without fear brow (Ekman p.148: fear brow hardest to voluntarily produce)
    if emotions["fear"] > 0.25:
        if features["inner_brow_raise"] < 0.01:
            deception_components.append(0.70)
            clues.append({
                "type": "MORPHOLOGY",
                "label": "Fear Without Fear Brow",
                "detail": "Fear indicators in lower face present without inner brow raise — Ekman: fear brow not an emblem/punctuator; its absence signals simulation",
                "severity": "HIGH"
            })

    # Anger: brow drawn but lower eyelid tension missing (Ekman p.149)
    if emotions["anger"] > 0.25:
        if features["brow_gap"] < 0.35 and features["lower_lid_tension"] < 0.12:
            deception_components.append(0.50)
            clues.append({
                "type": "MORPHOLOGY",
                "label": "Anger Without Eyelid Tension",
                "detail": "Brow drawn together (anger/determination emblem — easily produced) but lower eyelid tension absent; Ekman: only element missing in anger simulation",
                "severity": "MEDIUM"
            })

    # Sadness: lower face sad but no sad brow (Ekman p.150: sad brow hardest to fake)
    if emotions["sadness"] > 0.25:
        if features["inner_brow_raise"] < 0.005 and features["lip_corner_dir"] < 0:
            deception_components.append(0.60)
            clues.append({
                "type": "MORPHOLOGY",
                "label": "Sadness Without Sad Brow",
                "detail": "Lip corners pulled down but no inner brow elevation — Ekman: sad brow not an emblem; its absence is strongest clue sadness is simulated",
                "severity": "HIGH"
            })

    # ── 2. TIMING: Onset, duration, offset anomalies ──────────────
    # Ekman p.151: "timing is probably best source of deception clues about simulated surprise"

    if len(history_features) >= 6 and len(history_times) >= 6:
        # Onset speed: measure rate of change in dominant emotion features
        recent = list(history_features)[-6:]
        oldest_mouth = recent[0].get("mouth_width", 0)
        newest_mouth = recent[-1].get("mouth_width", 0)
        onset_speed = abs(newest_mouth - oldest_mouth)

        # Too-fast onset of broad smile = often posed
        if emotions["happiness"] > 0.4 and onset_speed > 0.08:
            deception_components.append(0.45)
            clues.append({
                "type": "TIMING",
                "label": "Abrupt Expression Onset",
                "detail": "Happiness expression appeared very rapidly; Ekman: genuine emotions build gradually — sharp onset suggests posed/managed expression",
                "severity": "MEDIUM"
            })

        # Prolonged surprise — Ekman p.148: surprise always brief; if prolonged, likely false
        if emotions["surprise"] > 0.30:
            surprise_duration = sum(
                1 for f in list(history_features)[-30:]
                if f.get("outer_brow_raise", 0) > 0.02 and f.get("jaw_drop", 0) > 0.01
            )
            if surprise_duration > 20:  # >~0.7s of surprise
                deception_components.append(0.70)
                clues.append({
                    "type": "TIMING",
                    "label": "Prolonged Surprise",
                    "detail": f"Surprise expression held for ~{surprise_duration/30:.1f}s — Ekman: surprise is always brief; prolonged surprise is almost certainly simulated",
                    "severity": "HIGH"
                })

    # ── 3. MICRO-EXPRESSION DETECTION ─────────────────────────────
    # Ekman p.723-725: micro-expressions result from interruptions; last 1/25 to 1/5 second
    # They reveal the emotion being concealed

    micro_exprs = []
    if len(history_features) >= 10:
        hist = list(history_features)
        # Scan last 10 frames for brief flashes (1–4 frames) of suppressed emotion
        for i in range(max(0, len(hist)-10), len(hist)-2):
            f_prev = hist[i-1] if i > 0 else hist[i]
            f_curr = hist[i]
            f_next = hist[i+1]

            # Flash of fear (inner brow raise) in otherwise neutral/happy face
            if (f_curr["inner_brow_raise"] > f_prev["inner_brow_raise"] * 1.8 and
                f_curr["inner_brow_raise"] > f_next["inner_brow_raise"] * 1.8 and
                f_curr["inner_brow_raise"] > 0.015):
                micro_exprs.append(("FEAR", i))

            # Flash of anger (brow draw) in otherwise neutral face
            if (f_curr["brow_gap"] < f_prev["brow_gap"] * 0.85 and
                f_curr["brow_gap"] < f_next["brow_gap"] * 0.85 and
                f_curr["brow_gap"] < 0.30):
                micro_exprs.append(("ANGER", i))

            # Flash of disgust (lip raise) in neutral face
            if (f_curr["upper_lip_raise"] > f_prev["upper_lip_raise"] * 1.5 and
                f_curr["upper_lip_raise"] > f_next["upper_lip_raise"] * 1.5 and
                f_curr["upper_lip_raise"] > 0.06):
                micro_exprs.append(("DISGUST", i))

    for emotion_type, frame_idx in micro_exprs[-3:]:  # report max 3
        deception_components.append(0.80)
        clues.append({
            "type": "MICRO-EXPRESSION",
            "label": f"Micro-Expression: {emotion_type}",
            "detail": f"Brief suppressed flash of {emotion_type.lower()} detected (duration ~1–4 frames) — Ekman: 'micro-expressions can reveal emotions the person is attempting to conceal'",
            "severity": "CRITICAL"
        })

    # ── 4. LEAKAGE: Felt emotion bleeding through managed expression ─
    # Ekman p.4344-4346: "leakage = nonintended betrayal of a feeling the person is trying to conceal"

    # Inner brow raises (sadness/fear) appearing with managed happy mouth
    if features["inner_brow_raise"] > 0.012 and emotions["happiness"] > 0.35:
        deception_components.append(0.60)
        clues.append({
            "type": "LEAKAGE",
            "label": "Fear/Sadness Brow With Happy Mouth",
            "detail": "Inner brow elevation (leakage of fear/sadness — hard to control) paired with managed happy lower face; Ekman: mouth managed, brow/forehead reveals true feeling",
            "severity": "HIGH"
        })

    # Anger leaking through disgust mask (Ekman p.149)
    if emotions["disgust"] > 0.25 and features["lower_lid_tension"] > 0.18:
        deception_components.append(0.50)
        clues.append({
            "type": "LEAKAGE",
            "label": "Anger Leaking Through Disgust",
            "detail": "Lower eyelid tension and hard stare (anger) present behind disgust expression — Ekman: anger may leak in stare when disgust is used as a mask",
            "severity": "MEDIUM"
        })

    # ── Compute final deception score ─────────────────────────────
    if deception_components:
        # Weighted average, boosted by quantity (multiple simultaneous clues = stronger signal)
        base_score = float(np.mean(deception_components))
        quantity_boost = min(len(deception_components) * 0.08, 0.25)
        deception_score = min(base_score + quantity_boost, 0.97)
    else:
        deception_score = 0.05 + np.random.uniform(0, 0.05)  # baseline noise

    return {
        "deception_score": round(float(deception_score), 3),
        "clues": clues,
        "dominant_emotion": dominant,
        "emotion_confidence": round(dominant_conf, 3),
    }


def analyse_voice_deception(af, history, baseline):
    """
    Vocal deception indicators — all features computed browser-side (Web Audio API).

    Indicators:
      1. Pitch elevation     — F0 rise above baseline (Vrij 2008; DePaulo 2003)
      2. Jitter elevation    — pitch-period instability above baseline level
      3. Pause rate          — more/longer silences than baseline (fabrication load)
      4. Speech rate change  — syllable rate deviation (slower=fabricating, faster=rehearsed)
      5. Vocal energy drop   — RMS well below baseline (suppressed/controlled affect)
      6. Cross-question      — this segment's profile vs session mean (strongest signal)
    """
    clues      = []
    components = []

    pitch         = af.get("pitch",         0.0)
    rms           = af.get("rms",           0.0)
    speaking      = af.get("speaking",      False)
    jitter        = af.get("jitter",        0.0)
    syllable_rate = af.get("syllable_rate", 0.0)
    pause_rate    = af.get("pause_rate",    0.0)

    if not speaking or pitch < 60:
        return {"voice_score": 0.0, "clues": [], "speaking": False}

    voiced = [h for h in list(history)[-20:]
              if h.get("speaking") and h.get("pitch", 0) > 60]

    # ── 1. Pitch elevation ────────────────────────────────
    if baseline.get("pitch", 0) > 60:
        ratio = pitch / baseline["pitch"]
        if ratio > 1.12:
            components.append(min((ratio - 1.0) * 4, 0.80))
            clues.append({
                "type":     "VOICE",
                "label":    "Elevated Pitch",
                "detail":   (f"Vocal F0 {int((ratio-1)*100)}% above baseline — pitch "
                             f"elevation under stress is one of the most replicated "
                             f"voice-deception markers (Vrij 2008; DePaulo et al. 2003)"),
                "severity": "HIGH" if ratio > 1.20 else "MEDIUM",
            })

    # ── 2. Jitter elevation ───────────────────────────────
    # Jitter = frame-to-frame pitch period variation / mean period.
    # Healthy speech: ~0.5–1.0%; stressed larynx: >2%
    if baseline.get("jitter", 0) > 0 and jitter > 0:
        jitter_ratio = jitter / baseline["jitter"]
        if jitter_ratio > 2.0:
            components.append(min((jitter_ratio - 1.0) * 0.25, 0.60))
            clues.append({
                "type":     "VOICE",
                "label":    "Elevated Vocal Jitter",
                "detail":   (f"Pitch period instability {int(jitter_ratio)}× above baseline — "
                             f"laryngeal micro-tremor indicates physiological stress even when "
                             f"mean pitch appears controlled"),
                "severity": "MEDIUM",
            })
    elif jitter > 0.025 and len(voiced) >= 4:
        # No baseline yet — flag absolute threshold
        components.append(0.30)
        clues.append({
            "type":     "VOICE",
            "label":    "High Vocal Jitter",
            "detail":   "Pitch period instability above clinical threshold — possible stress marker",
            "severity": "LOW",
        })

    # ── 3. Pause rate elevation ───────────────────────────
    # Count speaking→silent transitions in recent 20-frame window (~2.5s)
    if len(history) >= 10:
        window = list(history)[-20:]
        pauses_in_window = sum(
            1 for i in range(1, len(window))
            if window[i-1].get("speaking") and not window[i].get("speaking")
        )
        speech_frames = sum(1 for h in window if h.get("speaking"))
        speech_secs   = speech_frames * 0.12

        if speech_secs > 1.5:
            recent_pause_rate = pauses_in_window / speech_secs * 60  # pauses/min

            if baseline.get("pause_rate", 0) > 0:
                pr_ratio = recent_pause_rate / (baseline["pause_rate"] + 1e-6)
                if pr_ratio > 1.6:
                    components.append(min((pr_ratio - 1.0) * 0.30, 0.55))
                    clues.append({
                        "type":     "VOICE",
                        "label":    "Elevated Pause Frequency",
                        "detail":   (f"Pause rate {int(pr_ratio)}× above baseline — increased "
                                     f"hesitation is associated with the cognitive load of "
                                     f"fabricating responses (Vrij 2004)"),
                        "severity": "MEDIUM",
                    })
            elif recent_pause_rate > 25:
                # No baseline — flag high absolute rate
                components.append(0.25)
                clues.append({
                    "type":     "VOICE",
                    "label":    "Frequent Pausing",
                    "detail":   "High pause frequency during speech — possible cognitive load indicator",
                    "severity": "LOW",
                })

    # ── 4. Speech rate deviation ──────────────────────────
    if syllable_rate > 0 and baseline.get("syllable_rate", 0) > 0:
        sr_ratio = syllable_rate / baseline["syllable_rate"]
        if sr_ratio < 0.65:
            components.append(min((1.0 - sr_ratio) * 0.70, 0.55))
            clues.append({
                "type":     "VOICE",
                "label":    "Speech Rate Slowing",
                "detail":   (f"Syllable rate {int((1-sr_ratio)*100)}% below baseline — "
                             f"slowing is the most consistent speech-rate finding in "
                             f"deception research, reflecting increased fabrication effort"),
                "severity": "MEDIUM",
            })
        elif sr_ratio > 1.45:
            components.append(min((sr_ratio - 1.0) * 0.50, 0.45))
            clues.append({
                "type":     "VOICE",
                "label":    "Speech Rate Acceleration",
                "detail":   (f"Syllable rate {int((sr_ratio-1)*100)}% above baseline — "
                             f"acceleration can indicate rehearsed or scripted delivery"),
                "severity": "LOW",
            })

    # ── 5. Vocal energy drop ──────────────────────────────
    if baseline.get("rms", 0) > 0:
        rms_ratio = rms / (baseline["rms"] + 1e-6)
        if rms_ratio < 0.60:
            components.append(0.35)
            clues.append({
                "type":     "VOICE",
                "label":    "Reduced Vocal Energy",
                "detail":   "Vocal intensity significantly below baseline — suppressed affect or reduced effort",
                "severity": "LOW",
            })

    # ── 6. Cross-question comparison ──────────────────────
    # Compare this question's vocal profile against the session mean.
    # Only activates once ≥2 questions are marked and current question has data.
    if question_current >= 2 and len(question_voice_data) >= 2:
        # Build session means from all completed questions
        all_pitches  = []
        all_jitters  = []
        all_sr       = []
        for q, qd in question_voice_data.items():
            if q != question_current:
                all_pitches.extend(qd.get("pitches", []))
                all_jitters.extend(qd.get("jitters", []))
                all_sr.extend(qd.get("syllable_rates", []))

        if all_pitches and voiced:
            session_mean_pitch = float(np.mean(all_pitches))
            current_mean_pitch = float(np.mean([h["pitch"] for h in voiced]))
            if session_mean_pitch > 60:
                xq_ratio = current_mean_pitch / session_mean_pitch
                if xq_ratio > 1.15:
                    components.append(min((xq_ratio - 1.0) * 3.5, 0.75))
                    clues.append({
                        "type":     "VOICE",
                        "label":    f"Q{question_current}: Pitch Above Session Mean",
                        "detail":   (f"Pitch on this question is {int((xq_ratio-1)*100)}% "
                                     f"above the mean across other questions — cross-question "
                                     f"comparison is the strongest available voice indicator "
                                     f"as it controls for individual baseline"),
                        "severity": "HIGH",
                    })

        if all_jitters and jitter > 0:
            session_mean_jitter = float(np.mean(all_jitters))
            if session_mean_jitter > 0 and jitter / session_mean_jitter > 2.0:
                components.append(0.45)
                clues.append({
                    "type":     "VOICE",
                    "label":    f"Q{question_current}: Jitter Above Session Mean",
                    "detail":   "Vocal jitter on this question significantly exceeds other questions — strong stress indicator",
                    "severity": "HIGH",
                })

    voice_score = float(np.mean(components)) if components else 0.0
    return {
        "voice_score":    round(min(voice_score, 0.92), 3),
        "clues":          clues,
        "speaking":       True,
        "pitch":          round(pitch, 1),
        "rms":            round(rms, 4),
        "jitter":         round(jitter, 4),
        "syllable_rate":  round(syllable_rate, 2),
        "pause_rate":     round(pause_rate, 2),
    }


def compute_baseline(features_list):
    """Compute neutral baseline mean from frames. Returns mean dict."""
    if not features_list:
        return {}
    baseline = {}
    for key in features_list[0]:
        vals = [f[key] for f in features_list if key in f]
        baseline[key] = float(np.mean(vals)) if vals else 0.0
    return baseline


def compute_baseline_sigma(features_list):
    """Compute per-feature standard deviation from neutral frames.
    Used as personalised noise floor: 2.5σ replaces fixed gate constants."""
    if not features_list:
        return {}
    sigma = {}
    for key in features_list[0]:
        vals = [f[key] for f in features_list if key in f]
        sigma[key] = float(np.std(vals)) if len(vals) > 1 else 0.005
    return sigma


# ──────────────────────────────────────────────────────────
# Main analysis pipeline
# ──────────────────────────────────────────────────────────

def extract_forehead_rgb(frame, lms, w, h, iod):
    """
    Extract mean RGB from the forehead skin ROI.
    Uses landmark 9 (glabella) as centre and landmark 10 (upper forehead) as top.
    ROI width is 80% of IOD for good coverage without eyebrow contamination.
    Returns (r, g, b, g_norm) where g_norm = g / (r+g+b) cancels lighting drift.
    """
    try:
        cx  = int(lms[9].x  * w)
        cy  = int(lms[9].y  * h)
        ty  = int(lms[10].y * h)
        half = max(int(iod * 0.40), 8)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, ty),        min(h, cy)
        if x2 <= x1 or y2 <= y1 or (y2 - y1) < 4:
            return None
        roi   = frame[y1:y2, x1:x2]          # BGR in OpenCV
        mean_b = float(np.mean(roi[:, :, 0]))
        mean_g = float(np.mean(roi[:, :, 1]))
        mean_r = float(np.mean(roi[:, :, 2]))
        total  = mean_r + mean_g + mean_b + 1e-6
        return mean_r, mean_g, mean_b, mean_g / total
    except Exception:
        return None


def compute_hr_bpm(buf):
    """
    Estimate heart rate from rolling rPPG buffer using FFT.
    Algorithm:
      1. Detrend signal (subtract 3rd-order polynomial — removes slow drift)
      2. Apply Hanning window (reduces spectral leakage)
      3. Zero-padded FFT for sub-0.1 Hz frequency resolution
      4. Dominant frequency in cardiac band (0.7–3.5 Hz = 42–210 BPM)
    Returns (bpm, quality) where quality 0→1 indicates SNR in cardiac band.
    """
    if len(buf) < 40:
        return 0.0, 0.0

    entries  = list(buf)
    times    = np.array([e['t']      for e in entries])
    g_signal = np.array([e['g_norm'] for e in entries])

    # Estimate actual sampling rate from timestamps
    dts = np.diff(times)
    fs  = float(1.0 / np.mean(dts)) if len(dts) > 0 and np.mean(dts) > 0 else 8.0
    fs  = float(np.clip(fs, 3.0, 30.0))

    # Detrend: subtract fitted polynomial to remove slow lighting changes
    x      = np.linspace(0, 1, len(g_signal))
    coeffs = np.polyfit(x, g_signal, 3)
    signal = g_signal - np.polyval(coeffs, x)

    # Hanning window
    signal = signal * np.hanning(len(signal))

    # Zero-padded FFT (4× zero-padding improves frequency resolution)
    n_fft   = max(len(signal) * 4, 512)
    fft_mag = np.abs(np.fft.rfft(signal, n=n_fft))
    freqs   = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # Cardiac bandpass 0.7–3.5 Hz
    mask = (freqs >= 0.7) & (freqs <= 3.5)
    if not np.any(mask):
        return 0.0, 0.0

    cardiac_mag  = fft_mag[mask]
    cardiac_freq = freqs[mask]
    peak_idx     = int(np.argmax(cardiac_mag))
    peak_freq    = float(cardiac_freq[peak_idx])
    peak_mag     = float(cardiac_mag[peak_idx])
    mean_mag     = float(np.mean(cardiac_mag))

    # SNR-based quality: peak / mean in cardiac band
    # SNR ~1 = pure noise; SNR ~10+ = clean PPG signal
    snr     = peak_mag / (mean_mag + 1e-6)
    quality = float(np.clip((snr - 1.5) / 8.0, 0.0, 1.0))

    bpm = peak_freq * 60.0
    return round(float(bpm), 1), round(quality, 3)


def analyse_hr_deception(bpm, quality, history, baseline_bpm, current_deception_score):
    """
    Heart rate deception indicators.
    Only fires when signal quality is sufficient (>0.25) and baseline is set.
    Indicators:
      1. HR elevation above baseline  — autonomic arousal (hardest to fake)
      2. Rising HR trend              — acute stress response
      3. HR–face incongruence         — elevated HR with low facial deception score
    """
    clues      = []
    components = []

    if bpm <= 0 or quality < 0.25:
        return {"hr_score": 0.0, "clues": [], "bpm": bpm, "quality": quality}

    # ── 1. Elevation above baseline ───────────────────────
    if baseline_bpm > 0:
        elevation = bpm - baseline_bpm
        if elevation > 12:
            sev_val = float(np.clip(elevation / 35.0, 0, 0.85))
            components.append(sev_val)
            clues.append({
                "type":     "CARDIAC",
                "label":    "Elevated Heart Rate",
                "detail":   (f"HR {int(elevation)} BPM above resting baseline — autonomic "
                             f"arousal; heart rate is controlled by the involuntary nervous "
                             f"system and cannot be suppressed through facial management"),
                "severity": "HIGH" if elevation > 22 else "MEDIUM",
            })

    # ── 2. Rising trend ───────────────────────────────────
    recent_bpms = [h['bpm'] for h in list(history)[-12:] if h['bpm'] > 0]
    older_bpms  = [h['bpm'] for h in list(history)[-24:-12] if h['bpm'] > 0]
    if len(recent_bpms) >= 4 and len(older_bpms) >= 4:
        trend = float(np.mean(recent_bpms) - np.mean(older_bpms))
        if trend > 8:
            components.append(float(np.clip(trend / 25.0, 0, 0.60)))
            clues.append({
                "type":     "CARDIAC",
                "label":    "Rising Heart Rate",
                "detail":   (f"HR increasing {int(trend)} BPM over recent period — "
                             f"acute stress escalation consistent with mounting deceptive pressure"),
                "severity": "MEDIUM",
            })

    # ── 3. HR–face incongruence ───────────────────────────
    # High HR + low facial deception score = controlled presentation
    if baseline_bpm > 0 and (bpm - baseline_bpm) > 15 and current_deception_score < 0.30:
        components.append(0.65)
        clues.append({
            "type":     "CARDIAC",
            "label":    "HR–Face Incongruence",
            "detail":   ("Significantly elevated heart rate paired with controlled facial "
                         "expression — indicates active suppression of visible stress signals; "
                         "this combination is a strong deception indicator in skilled liars"),
            "severity": "HIGH",
        })

    hr_score = float(np.mean(components)) if components else 0.0
    return {
        "hr_score": round(min(hr_score, 0.90), 3),
        "clues":    clues,
        "bpm":      bpm,
        "quality":  quality,
        "elevation": round(bpm - baseline_bpm, 1) if baseline_bpm > 0 else 0,
    }


def process_frame(img_bytes, audio_features=None):
    global frame_count, baseline_features, emotion_ema

    # Decode image
    nparr  = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
    frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    h, w = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb   = np.ascontiguousarray(rgb)   # MediaPipe requires contiguous memory

    # Run MediaPipe (Tasks API: wrap ndarray in mp.Image, unpack face_landmarks)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = get_landmarker().detect(mp_image)
    if not detection.face_landmarks:
        return {"error": "no_face", "message": "No face detected"}

    lms      = detection.face_landmarks[0]   # list of NormalizedLandmark (same .x/.y/.z)
    features = extract_features(lms, w, h)

    # ── rPPG: extract forehead colour signal ──────────────
    # Wrapped in try/except — rPPG must never prevent face analysis from completing
    hr_result = {"bpm": 0.0, "quality": 0.0, "hr_score": 0.0, "clues": [], "elevation": 0}
    try:
        iod_dist = max(math.dist(
            [lms[33].x * w,  lms[33].y * h],
            [lms[263].x * w, lms[263].y * h]
        ), 1.0)
        rgb_result = extract_forehead_rgb(frame, lms, w, h, iod_dist)
        if rgb_result is not None:
            _, _, _, g_norm = rgb_result
            rppg_buffer.append({"t": time.time(), "g_norm": g_norm})
            bpm, quality = compute_hr_bpm(rppg_buffer)
            if bpm > 0:
                hr_history.append({"t": time.time(), "bpm": bpm, "quality": quality})
    except Exception as e:
        print(f"[VERITY] rPPG error (non-fatal): {e}", flush=True)

    # Store history
    now = time.time()
    expression_history.append(features)
    timestamp_history.append(now)
    frame_count += 1

    # Baseline is now set manually via POST /baseline — no auto-capture

    # Adjust features relative to baseline.
    # Emotion classification uses delta features once calibration is done —
    # classify_emotion(is_delta=True) adapts its formulas accordingly.
    # Deception analysis also uses delta features (checks for deviations from neutral).
    adj_features = features.copy()
    if baseline_features:
        for k in adj_features:
            if k in baseline_features:
                adj_features[k] = features[k] - baseline_features[k]

    calibrated   = bool(baseline_features)
    raw_emotions = classify_emotion(adj_features if calibrated else features,
                                    is_delta=calibrated)

    # EMA smoothing: reduces frame-to-frame jitter without adding perceptible lag.
    # α=0.35 → ~2-frame time constant at 30fps — fast enough to track expressions,
    # slow enough to stop the bars from flickering on every micro-movement.
    if emotion_ema:
        for k in raw_emotions:
            emotion_ema[k] = EMA_ALPHA * raw_emotions[k] + (1.0 - EMA_ALPHA) * emotion_ema.get(k, raw_emotions[k])
    else:
        emotion_ema = raw_emotions.copy()

    emotions = {k: round(emotion_ema[k], 3) for k in emotion_ema}

    # Deception analysis
    deception = analyse_deception(
        adj_features if baseline_features else features,
        emotions,
        list(expression_history),
        list(timestamp_history)
    )

    # Face region highlights for overlay
    def region_pts(indices):
        return [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices if i < len(lms)]

    regions = {
        "brow_left":   region_pts(LANDMARKS["inner_brow_left"]  + LANDMARKS["outer_brow_left"]),
        "brow_right":  region_pts(LANDMARKS["inner_brow_right"] + LANDMARKS["outer_brow_right"]),
        "eye_left":    region_pts(LANDMARKS["upper_lid_left"]   + LANDMARKS["lower_lid_left"]),
        "eye_right":   region_pts(LANDMARKS["upper_lid_right"]  + LANDMARKS["lower_lid_right"]),
        "mouth":       region_pts(LANDMARKS["mouth_corner_left"] + LANDMARKS["mouth_corner_right"] +
                                  LANDMARKS["upper_lip"][:5] + LANDMARKS["lower_lip"][:5]),
        "nose":        region_pts(LANDMARKS["nose_bridge"] + LANDMARKS["nose_lower"]),
    }

    elapsed = round(now - session_start, 1)

    # ── Voice deception analysis ───────────────────────────
    voice_result = {"voice_score": 0.0, "clues": [], "speaking": False}
    if audio_features:
        voice_history.append(audio_features)
        voice_result = analyse_voice_deception(
            audio_features, list(voice_history), voice_baseline
        )
        # Merge voice clues into the main deception result
        if voice_result.get("clues"):
            deception["clues"] = deception.get("clues", []) + voice_result["clues"]
            f_score = deception["deception_score"]
            v_score = voice_result["voice_score"]
            combined = 0.60 * f_score + 0.40 * v_score
            quantity_bonus = min(len(voice_result["clues"]) * 0.04, 0.12)
            deception["deception_score"] = round(min(combined + quantity_bonus, 0.97), 3)

    # ── HR deception analysis ──────────────────────────────
    try:
        if rppg_buffer and hr_history:
            latest = hr_history[-1]
            hr_result = analyse_hr_deception(
                latest['bpm'], latest['quality'],
                list(hr_history), hr_baseline,
                deception['deception_score']
            )
            if hr_result.get('clues'):
                deception['clues'] = deception.get('clues', []) + hr_result['clues']
                hr_score = hr_result['hr_score']
                existing = deception['deception_score']
                combined = 0.55 * existing + 0.15 * (voice_result.get('voice_score',0)) + 0.30 * hr_score
                quantity_bonus = min(len(hr_result['clues']) * 0.05, 0.12)
                deception['deception_score'] = round(min(combined + quantity_bonus, 0.97), 3)
    except Exception as e:
        print(f"[VERITY] HR analysis error (non-fatal): {e}", flush=True)

    # ── Accumulate per-question voice stats ───────────────
    if question_current > 0 and audio_features and audio_features.get("speaking"):
        if question_current not in question_voice_data:
            question_voice_data[question_current] = {
                "scores": [], "clues": [], "pitches": [],
                "jitters": [], "syllable_rates": []
            }
        qd = question_voice_data[question_current]
        qd["scores"].append(deception["deception_score"])
        if voice_result.get("clues"):
            qd["clues"].extend(voice_result["clues"])
        if audio_features.get("pitch", 0) > 60:
            qd["pitches"].append(audio_features["pitch"])
        if audio_features.get("jitter", 0) > 0:
            qd["jitters"].append(audio_features["jitter"])
        if audio_features.get("syllable_rate", 0) > 0:
            qd["syllable_rates"].append(audio_features["syllable_rate"])

    # Build question summary for the frontend
    q_summaries = {}
    for q, qd in question_voice_data.items():
        if qd["scores"]:
            q_summaries[q] = {
                "avg": round(float(np.mean(qd["scores"])) * 100, 1),
                "max": round(float(np.max(qd["scores"]))  * 100, 1),
                "n_clues": len(set(c["label"] for c in qd["clues"])),
                "n_frames": len(qd["scores"]),
            }

    return {
        "status":              "ok",
        "frame":               frame_count,
        "elapsed":             elapsed,
        "features":            features,
        "emotions":            emotions,
        "deception":           deception,
        "regions":             regions,
        "voice":               voice_result,
        "hr":                  hr_result,
        "baseline_ready":      bool(baseline_features),
        "calibrating":         frame_count < 35,
        "calibration_exprs":   sorted(feature_peaks.keys()),
        "question_current":    question_current,
        "question_summaries":  q_summaries,
        "question_markers":    question_markers[-20:],
    }


# ──────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    img_b64       = data["image"].split(",")[-1]
    audio_features = data.get("audio")          # optional — None if mic not available
    result        = process_frame(img_b64, audio_features)
    return jsonify(result)


@app.route("/mark_question", methods=["POST"])
def mark_question():
    """Mark the start of a new question for cross-question voice comparison."""
    global question_current, question_markers
    question_current += 1
    question_markers.append({
        "question": question_current,
        "elapsed":  round(time.time() - session_start, 1),
        "frame":    frame_count,
    })
    # Initialise accumulator for this question
    question_voice_data[question_current] = {
        "scores": [], "clues": [], "pitches": [],
        "jitters": [], "syllable_rates": []
    }
    return jsonify({"status": "ok", "question": question_current})


@app.route("/reset", methods=["POST"])
def reset():
    global frame_count, baseline_features, session_start, emotion_ema
    global voice_history, voice_baseline, baseline_sigma, feature_peaks
    global question_current, question_markers, question_voice_data
    global rppg_buffer, hr_history, hr_baseline
    expression_history.clear()
    timestamp_history.clear()
    voice_history.clear()
    rppg_buffer.clear()
    hr_history.clear()
    baseline_features  = {}
    baseline_sigma     = {}
    feature_peaks      = {}
    voice_baseline     = {}
    emotion_ema        = {}
    hr_baseline        = 0.0
    question_current   = 0
    question_markers   = []
    question_voice_data = {}
    frame_count       = 0
    session_start     = time.time()
    return jsonify({"status": "reset"})


@app.route("/baseline", methods=["POST"])
def set_baseline():
    """
    Snapshot the current neutral expression as the personal baseline.
    Uses the last 30 frames (min 3). Called manually by the user.
    """
    global baseline_features, emotion_ema, voice_baseline
    try:
        n = len(expression_history)
        if n < 3:
            return jsonify({
                "error": f"Face not detected long enough ({n} frames) — keep face in view and try again"
            }), 400

        recent            = list(expression_history)[-30:]
        baseline_features = compute_baseline(recent)
        baseline_sigma    = compute_baseline_sigma(recent)   # per-person noise floors
        emotion_ema       = {}

        # Capture voice baseline from recent voiced frames (optional)
        voiced = [v for v in list(voice_history)[-20:]
                  if v.get("speaking") and v.get("pitch", 0) > 60]
        if voiced:
            voice_baseline = {
                "pitch":         float(np.mean([v["pitch"] for v in voiced])),
                "rms":           float(np.mean([v["rms"]   for v in voiced])),
                "jitter":        float(np.mean([v.get("jitter", 0) for v in voiced])),
                "syllable_rate": float(np.mean([v["syllable_rate"] for v in voiced
                                                if v.get("syllable_rate", 0) > 0]) or 0),
                "pause_rate":    float(np.mean([v.get("pause_rate", 0) for v in voiced])),
            }

        # Capture resting heart rate from recent clean readings
        recent_hr = [h['bpm'] for h in list(hr_history)[-20:]
                     if h['bpm'] > 0 and h.get('quality', 0) > 0.30]
        if recent_hr:
            hr_baseline = float(np.mean(recent_hr))

        return jsonify({
            "status":         "ok",
            "frames_used":    len(recent),
            "voice_baseline": bool(voice_baseline),
            "hr_baseline":    round(hr_baseline, 1),
        })

    except Exception as exc:
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/calibrate", methods=["POST"])
def calibrate_expression():
    """
    Capture the peak feature values for one calibration expression.
    Called once per expression after the user has been holding it for ~3 seconds.
    expression: 'eyebrows' | 'smile' | 'anger'
    Stores raw feature means in feature_peaks[expression] for use in
    classify_emotion's peak-normalisation path.
    """
    global feature_peaks
    try:
        data = request.get_json() or {}
        expr = data.get("expression", "")
        if expr not in ("eyebrows", "smile", "anger"):
            return jsonify({"error": f"Unknown expression: {expr}"}), 400

        if len(expression_history) < 3:
            return jsonify({"error": "Not enough frames — keep face in view"}), 400

        recent = list(expression_history)[-20:]   # last ~2.5s
        peaks  = {k: float(np.mean([fr[k] for fr in recent if k in fr]))
                  for k in recent[0]}
        feature_peaks[expr] = peaks

        return jsonify({
            "status":     "ok",
            "expression": expr,
            "captured":   len(recent),
            "complete":   sorted(feature_peaks.keys()),
        })

    except Exception as exc:
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/ai_analysis", methods=["POST"])
def ai_analysis():
    """
    Aggregate session data and call the Anthropic API to generate
    a forensic deception narrative. Requires ANTHROPIC_API_KEY env var.
    """
    import urllib.request as _ur
    import json as _json

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return jsonify({
            "status":  "error",
            "message": "ANTHROPIC_API_KEY environment variable not set — add it in your Render dashboard under Environment.",
        })

    try:
        data    = request.get_json() or {}
        history = data.get("history", [])

        if len(history) < 5:
            return jsonify({"status": "error", "message": "Not enough data — collect more session data first."})

        # ── Aggregate session statistics ───────────────────────────
        scores  = [h.get("deception_score", 0) for h in history]
        elapsed = [h.get("elapsed", 0)         for h in history]
        duration = max(elapsed) if elapsed else 0

        # Trend: split into thirds and compare averages
        n = len(scores)
        t1 = float(np.mean(scores[:n//3]))      if n >= 3 else 0
        t2 = float(np.mean(scores[n//3:2*n//3])) if n >= 3 else 0
        t3 = float(np.mean(scores[2*n//3:]))    if n >= 3 else 0
        trend = "escalating" if t3 > t1 + 0.08 else ("de-escalating" if t1 > t3 + 0.08 else "stable")

        # Emotion dominance across session
        emotion_totals = {}
        for h in history:
            for em, val in (h.get("emotions") or {}).items():
                emotion_totals[em] = emotion_totals.get(em, 0) + val
        dominant_emotion = max(emotion_totals, key=emotion_totals.get) if emotion_totals else "—"

        # Clue frequency
        clue_freq: dict = {}
        voice_clue_freq: dict = {}
        for h in history:
            for c in (h.get("clues") or []):
                label = c.get("label", "")
                if c.get("type") == "VOICE":
                    voice_clue_freq[label] = voice_clue_freq.get(label, 0) + 1
                else:
                    clue_freq[label] = clue_freq.get(label, 0) + 1

        top_clues = sorted(clue_freq.items(), key=lambda x: -x[1])[:5]
        top_voice = sorted(voice_clue_freq.items(), key=lambda x: -x[1])[:3]

        # Peak moments (top 3 frames by score)
        peaks = sorted(enumerate(scores), key=lambda x: -x[1])[:3]
        peak_strs = [f"{elapsed[i]:.0f}s ({scores[i]*100:.0f}%)" for i, _ in peaks if i < len(elapsed)]

        # ── Build prompt ───────────────────────────────────────────
        clue_text  = "; ".join(f"{l} ({n}×)" for l, n in top_clues)  or "None detected"
        voice_text = "; ".join(f"{l} ({n}×)" for l, n in top_voice) or "None detected"

        prompt = f"""You are a forensic deception analyst reviewing output from VERITY, a real-time facial and vocal deception detection system based on Paul Ekman & Wallace Friesen's Unmasking the Face (2003).

SESSION SUMMARY
Duration: {duration:.0f}s  |  Samples: {n}
Deception score — mean: {np.mean(scores)*100:.1f}%  max: {max(scores)*100:.1f}%  trend: {trend}
Early period avg: {t1*100:.1f}%  →  Mid: {t2*100:.1f}%  →  Late: {t3*100:.1f}%
Dominant emotion detected: {dominant_emotion}

FACIAL DECEPTION CLUES (by frequency)
{clue_text}

VOICE DECEPTION CLUES (by frequency)
{voice_text}

PEAK DECEPTION MOMENTS
{', '.join(peak_strs) if peak_strs else 'None above threshold'}

Return ONLY a valid JSON object — no markdown, no preamble, no explanation outside the JSON. Use this exact schema:

{{
  "verdict": "TRUTHFUL" | "INCONCLUSIVE" | "DECEPTION LIKELY",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "narrative": "3-4 sentences interpreting the overall pattern, trend, and most significant clues in professional forensic language.",
  "key_findings": ["concise finding 1", "concise finding 2", "concise finding 3"],
  "voice_summary": "1-2 sentences on voice indicators and their significance, or null if no voice data.",
  "examiner_note": "One actionable sentence for the examiner on what to probe next or what to treat with caution."
}}

Be precise but do not overclaim certainty — no system can determine deception with absolute confidence."""

        # ── Call Anthropic API ─────────────────────────────────────
        payload = _json.dumps({
            "model":      "claude-sonnet-4-6",
            "max_tokens": 600,
            "messages":   [{"role": "user", "content": prompt}],
        }).encode()

        req = _ur.Request(
            "https://api.anthropic.com/v1/messages",
            data    = payload,
            headers = {
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            method = "POST",
        )
        with _ur.urlopen(req, timeout=30) as resp:
            result   = _json.loads(resp.read())
            raw_text = result["content"][0]["text"].strip()

        # Strip any accidental markdown fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        try:
            report = _json.loads(raw_text)
        except Exception:
            # Fallback: return raw text if JSON parsing fails
            return jsonify({"status": "ok", "report": None, "raw": raw_text})

        # Attach session stats so the frontend can render the metrics row
        report["stats"] = {
            "mean":     round(float(np.mean(scores)) * 100, 1),
            "peak":     round(float(np.max(scores))  * 100, 1),
            "trend":    trend,
            "duration": round(duration),
            "samples":  n,
            "dominant": dominant_emotion,
            "top_clues":  [l for l, _ in top_clues],
            "top_voice":  [l for l, _ in top_voice],
        }
        return jsonify({"status": "ok", "report": report})

    except Exception as exc:
        return jsonify({"status": "error", "message": f"Analysis failed: {exc}"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":           "ok",
        "mediapipe":        "loaded",
        "api_key_set":      bool(os.environ.get("ANTHROPIC_API_KEY")),
        "api_key_length":   len(os.environ.get("ANTHROPIC_API_KEY", "")),
    })


@app.route("/analyse_video", methods=["POST"])
def analyse_video():
    """
    Accept an uploaded video file, process every Nth frame through the
    full Ekman pipeline, and return a complete timeline + summary report.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file (OpenCV needs a file path)
    suffix = ".webm"
    if "mp4" in video_file.content_type:
        suffix = ".mp4"
    elif "ogg" in video_file.content_type:
        suffix = ".ogv"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    try:
        result = process_video_file(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return jsonify(result)


def process_video_file(path):
    """
    Process a video file frame by frame.
    Analyses every 3rd frame to balance speed vs resolution.
    Returns full timeline + aggregated report.
    """
    # Dedicated landmarker instance for video — IMAGE mode is stateless,
    # safe to use per-frame without timestamp bookkeeping.
    video_landmarker = _make_landmarker(detection_conf=0.45, tracking_conf=0.45)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps_src    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps_src

    # Sample every 3rd frame
    SAMPLE_EVERY = 3
    timeline      = []   # per-sample results
    all_features  = []
    all_times     = []
    video_baseline = {}
    v_frame_count  = 0
    sample_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        v_frame_count += 1
        if v_frame_count % SAMPLE_EVERY != 0:
            continue

        sample_count += 1
        timestamp = v_frame_count / fps_src

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = video_landmarker.detect(mp_image)

        if not detection.face_landmarks:
            timeline.append({
                "t": round(timestamp, 2),
                "frame": v_frame_count,
                "face": False,
            })
            continue

        lms      = detection.face_landmarks[0]
        features = extract_features(lms, w, h)
        emotions = classify_emotion(features)

        all_features.append(features)
        all_times.append(timestamp)

        # Build baseline from first 30 detected-face samples
        if len(all_features) == 30:
            video_baseline = compute_baseline(all_features[:30])

        adj_features = features.copy()
        if video_baseline:
            for k in adj_features:
                if k in video_baseline:
                    adj_features[k] = features[k] - video_baseline[k]

        deception = analyse_deception(
            adj_features if video_baseline else features,
            emotions,
            all_features,
            all_times,
        )

        # Capture thumbnail (small JPEG) for key moments
        thumb_b64 = None
        if deception["deception_score"] > 0.55 or deception["clues"]:
            small = cv2.resize(frame, (120, 90))
            _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

        timeline.append({
            "t":              round(timestamp, 2),
            "frame":          v_frame_count,
            "face":           True,
            "emotions":       emotions,
            "features":       {k: round(v, 3) for k, v in features.items()},
            "deception_score": round(deception["deception_score"], 3),
            "clues":          deception["clues"],
            "dominant":       deception["dominant_emotion"],
            "thumb":          thumb_b64,
        })

    cap.release()
    video_landmarker.close()

    if not timeline:
        return {"error": "No frames processed"}

    # ── Aggregate summary ──────────────────────────────────
    face_samples = [s for s in timeline if s.get("face")]
    scores       = [s["deception_score"] for s in face_samples]
    all_clues    = []
    for s in face_samples:
        for c in s.get("clues", []):
            all_clues.append({**c, "t": s["t"]})

    # Key moments: top 5 highest deception score samples that have clues
    key_moments = sorted(
        [s for s in face_samples if s.get("clues") and s.get("thumb")],
        key=lambda x: x["deception_score"],
        reverse=True
    )[:5]

    # Deep-copy key moments NOW before we strip thumbs from timeline below
    import copy
    key_moments = [copy.deepcopy(m) for m in key_moments]

    # Clue frequency
    clue_freq = {}
    for c in all_clues:
        label = c["label"]
        clue_freq[label] = clue_freq.get(label, 0) + 1

    summary = {
        "duration":        round(duration, 1),
        "total_frames":    total_frames,
        "samples_analysed": len(face_samples),
        "face_coverage":   round(len(face_samples) / max(len(timeline), 1) * 100, 1),
        "avg_deception":   round(float(np.mean(scores)) * 100, 1) if scores else 0,
        "max_deception":   round(float(np.max(scores)) * 100, 1) if scores else 0,
        "verdict":         _verdict(float(np.mean(scores)) if scores else 0),
        "clue_freq":       dict(sorted(clue_freq.items(), key=lambda x: -x[1])),
        "key_moments":     key_moments,
    }

    # Strip thumbs from main timeline (keep only key moments)
    for s in timeline:
        s.pop("thumb", None)

    return {
        "status":   "ok",
        "summary":  summary,
        "timeline": timeline,
    }


def _verdict(score):
    if score < 0.25:
        return "TRUTHFUL"
    elif score < 0.55:
        return "INCONCLUSIVE"
    else:
        return "DECEPTION LIKELY"


if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  DECEPTION DETECTION ENGINE — Ekman/Friesen Framework")
    print("  Based on: Unmasking the Face (2003)")
    print("  Server: http://localhost:5050")
    print("═"*60 + "\n")
    app.run(debug=False, port=5050, host="0.0.0.0", threaded=True)
