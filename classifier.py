"""
VERITY — Emotion Classifier (HSEmotions ONNX backend)
======================================================
Uses HSEmotions ONNX — EfficientNet-B0 pretrained on VGAFer/AffectNet.
~15 MB model, no TensorFlow required, ~74% accuracy vs FER's ~65%.

Replaces the FER/TensorFlow classifier entirely.
Drop this file next to app.py — no training required.

Emotions returned by HSEmotions:
  Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
  → mapped to VERITY labels (Contempt and Neutral dropped from softmax)
"""

import numpy as np

# ── Label mapping: HSEmotions → VERITY ────────────────────────
# HSEmotions returns 8 classes in this fixed order:
HS_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear',
             'Happiness', 'Neutral', 'Sadness', 'Surprise']

HS_TO_VERITY = {
    'Anger':     'anger',
    'Disgust':   'disgust',
    'Fear':      'fear',
    'Happiness': 'happiness',
    'Sadness':   'sadness',
    'Surprise':  'surprise',
    # Contempt and Neutral are excluded from the emotion distribution
    # Neutral suppresses scores; Contempt isn't in Ekman's 6 basic emotions
}

# ── Lazy-load recogniser (initialised once on first call) ──────
_recogniser = None


def _get_recogniser():
    global _recogniser
    if _recogniser is None:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        # enet_b0_8_best_vgaf: best for real-world / in-the-wild faces
        # Downloads ~15 MB ONNX model on first call, cached automatically
        _recogniser = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
        print("[classifier] ✓ HSEmotions ONNX recogniser initialised", flush=True)
    return _recogniser


# ──────────────────────────────────────────────────────────────
# Primary entry point called by app.py
# ──────────────────────────────────────────────────────────────

def classify_emotion_frame(face_crop_bgr):
    """
    Run HSEmotions CNN on a face crop (BGR).
    Returns dict of {emotion: confidence 0-1} normalised to sum = 1.
    Falls back to uniform distribution if inference fails.

    Parameters
    ----------
    face_crop_bgr : np.ndarray  HxWx3 BGR image
                    Should be a tight face crop for best accuracy.
                    app.py extracts this from MediaPipe landmark bbox.
    """
    try:
        recogniser = _get_recogniser()

        # HSEmotions expects RGB
        import cv2
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        # predict_emotions returns (emotion_probabilities, dominant_emotion_str)
        _, probs = recogniser.predict_emotions(rgb, logits=False)

        # Map to VERITY labels, drop Neutral and Contempt
        scores = {}
        for label, prob in zip(HS_LABELS, probs):
            verity_label = HS_TO_VERITY.get(label)
            if verity_label:
                scores[verity_label] = float(prob)

        # Renormalise over the 6 kept emotions
        total = sum(scores.values()) + 1e-9
        return {k: round(v / total, 3) for k, v in scores.items()}

    except Exception as e:
        print(f"[classifier] HSEmotions error: {e} — using rule-based fallback",
              flush=True)
        return _uniform_fallback()


def _uniform_fallback():
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    p = round(1 / len(emotions), 3)
    return {e: p for e in emotions}


# ──────────────────────────────────────────────────────────────
# Legacy shim — keeps classify_emotion(features) calls working
# ──────────────────────────────────────────────────────────────

def classify_emotion(features):
    """Legacy interface — used when no frame is available."""
    return _rule_classify(features)


def _rule_classify(f):
    """Calibrated rule-based fallback (used only if HSEmotions fails)."""
    scores = {}

    duchenne = min(f.get("cheek_raise", 0), 0.08) / 0.08
    scores["happiness"] = float(np.clip(
        0.35 * max(f.get("lip_corner_dir", 0) - 0.02, 0) +
        0.35 * max(f.get("cheek_raise", 0), 0) +
        0.30 * f.get("lower_lid_tension", 0) * duchenne,
        0, 1))

    scores["surprise"] = float(np.clip(
        0.35 * max(f.get("outer_brow_raise", 0) - 0.01, 0) +
        0.30 * max(f.get("eye_aperture", 0) - 0.15, 0) +
        0.35 * max(f.get("jaw_drop", 0) - 0.02, 0),
        0, 1))

    scores["fear"] = float(np.clip(
        0.40 * max(f.get("inner_brow_raise", 0) - 0.005, 0) +
        0.25 * max(f.get("eye_aperture", 0) - 0.17, 0) +
        0.20 * max(f.get("mouth_open", 0) - 0.03, 0) +
        0.15 * max(f.get("mouth_width", 0) - 0.55, 0),
        0, 1))

    brow_draw = max(0.32 - f.get("brow_gap", 0.4), 0) / 0.32
    scores["anger"] = float(np.clip(
        0.40 * brow_draw +
        0.35 * max(f.get("lower_lid_tension", 0) - 0.12, 0) +
        0.25 * max(f.get("lip_press", 0) - 0.20, 0),
        0, 1))

    scores["disgust"] = float(np.clip(
        0.45 * max(f.get("upper_lip_raise", 0) - 0.08, 0) * 4 +
        0.40 * max(f.get("nose_wrinkle", 0) - 0.10, 0) * 3 +
        0.15 * max(-f.get("lip_corner_dir", 0) - 0.01, 0),
        0, 1))

    sad_brow = max(f.get("inner_brow_raise", 0) - 0.005, 0)
    sad_lip  = max(-f.get("lip_corner_dir", 0) - 0.01, 0)
    gate     = min(sad_brow * 15, 1.0) * min(sad_lip * 10, 1.0)
    scores["sadness"] = float(np.clip(
        gate * (0.45 * sad_brow * 10 + 0.35 * sad_lip * 10 +
                0.20 * max(-f.get("cheek_raise", 0) - 0.005, 0) * 10),
        0, 1))

    total = sum(scores.values()) + 1e-9
    return {k: round(float(v / total), 3) for k, v in scores.items()}
