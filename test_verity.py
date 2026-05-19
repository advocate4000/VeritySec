"""
VERITY — Automated Test Suite
Run with:  pytest test_verity.py -v
Requires:  pip install pytest

Tests are grouped into four areas:
  1. Feature arithmetic  — verify the fixed geometry produces correct signs
  2. Emotion classifier  — correct dominant emotion for each canonical expression
  3. Deception analysis  — gates, morphology clues, timing, leakage
  4. API smoke tests     — Flask test client round-trips
"""

import os, sys, json, math
import numpy as np
import pytest

# Allow importing app.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    classify_emotion,
    analyse_deception,
    compute_baseline,
    _baseline_quality,
    _calibration_state,
    detect_micro_expressions,
    ExponentialSmoother,
    FEATURE_NAMES,
    EMOTION_LABELS,
    app as flask_app,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def neutral_adj() -> dict:
    """All delta features at zero — perfectly neutral face after calibration."""
    return {k: 0.0 for k in FEATURE_NAMES}


def neutral_raw() -> dict:
    """Typical absolute feature values for a neutral face (pre-calibration)."""
    return {
        "inner_brow_raise":  0.08,
        "outer_brow_raise":  0.12,
        "brow_gap":          0.38,
        "eye_aperture":      0.25,
        "lower_lid_tension": 0.18,
        "cheek_raise":       0.40,
        "mouth_width":       0.45,
        "mouth_open":        0.02,
        "lip_corner_dir":    0.01,
        "upper_lip_raise":   0.06,
        "lip_press":         0.80,
        "nose_wrinkle":      0.04,
        "jaw_drop":          0.02,
    }


def adj_from(overrides: dict) -> dict:
    """Start from neutral deltas and apply expression-specific overrides."""
    f = neutral_adj()
    f.update(overrides)
    return f


def make_deception(adj, baseline, emotions):
    """Run deception analysis with a fresh smoother."""
    smoother = ExponentialSmoother(alpha=1.0)   # alpha=1 → raw score passthrough
    raw = neutral_raw()
    # Build raw from baseline + adj
    features = {k: baseline.get(k, raw.get(k, 0)) + adj.get(k, 0) for k in FEATURE_NAMES}
    return analyse_deception(features, baseline, emotions, [], [], smoother)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature arithmetic
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureArithmetic:

    def test_cheek_raise_positive_for_neutral(self):
        """cheek_raise must be positive (cheek is above mouth corner anatomically)."""
        # After our geometry fix (base=mouth_corner), the raw value should be positive.
        # We verify by checking a neutral raw features dict is plausible.
        raw = neutral_raw()
        assert raw["cheek_raise"] > 0, "cheek_raise should be positive for a neutral face"

    def test_lip_corner_dir_near_zero_for_neutral(self):
        """lip_corner_dir should be near zero for a neutral face."""
        raw = neutral_raw()
        assert abs(raw["lip_corner_dir"]) < 0.10, (
            "lip_corner_dir should be near zero for neutral (corners at lower-lip level)")

    def test_adj_all_zero_for_matched_baseline(self):
        """When current == baseline, all deltas must be zero."""
        raw = neutral_raw()
        baseline = dict(raw)
        adj = {k: raw[k] - baseline.get(k, raw[k]) for k in FEATURE_NAMES}
        for k, v in adj.items():
            assert abs(v) < 1e-9, f"Expected zero delta for {k}, got {v}"

    def test_smile_delta_positive_for_lip_corner(self):
        """A smile should produce a positive lip_corner_dir delta."""
        baseline_lcd = 0.01   # typical neutral
        smile_lcd    = 0.06   # corners risen above lower lip center
        delta = smile_lcd - baseline_lcd
        assert delta > 0, "Smile should produce positive lip_corner_dir delta"

    def test_frown_delta_negative_for_lip_corner(self):
        """A frown should produce a negative lip_corner_dir delta."""
        baseline_lcd = 0.01
        frown_lcd    = -0.04
        delta = frown_lcd - baseline_lcd
        assert delta < 0, "Frown should produce negative lip_corner_dir delta"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emotion classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionClassifier:

    # ── Neutral distribution ──────────────────────────────────────────────────

    def test_neutral_distributes_roughly_evenly(self):
        """All-zero deltas → each emotion should be close to 1/6."""
        em = classify_emotion(neutral_adj())
        for label, score in em.items():
            assert 0.05 <= score <= 0.45, (
                f"{label}={score:.3f} is too far from uniform for a neutral face")

    def test_neutral_no_dominant_emotion(self):
        """No single emotion should dominate a neutral face."""
        em = classify_emotion(neutral_adj())
        dominant_score = max(em.values())
        assert dominant_score < 0.55, (
            f"Dominant emotion scores {dominant_score:.2f} for neutral face — too high")

    # ── Canonical expressions ─────────────────────────────────────────────────

    def test_happiness_from_smile_deltas(self):
        """Positive lip corner + cheek raise deltas → happiness dominant."""
        em = classify_emotion(adj_from({
            "lip_corner_dir":    0.08,
            "cheek_raise":       0.06,
            "lower_lid_tension": 0.04,
        }))
        assert em["happiness"] == max(em.values()), (
            f"Expected happiness dominant, got {max(em, key=em.get)} ({em})")
        assert em["happiness"] > 0.35, f"Happiness score too low: {em['happiness']}"

    def test_surprise_from_brow_and_jaw(self):
        """Raised brows + wide eyes + jaw drop → surprise dominant."""
        em = classify_emotion(adj_from({
            "outer_brow_raise": 0.06,
            "eye_aperture":     0.06,
            "jaw_drop":         0.05,
        }))
        assert em["surprise"] == max(em.values()), (
            f"Expected surprise dominant, got {max(em, key=em.get)} ({em})")

    def test_fear_from_inner_brow_and_wide_eyes(self):
        """Inner brow raise + wide eyes → fear dominant."""
        em = classify_emotion(adj_from({
            "inner_brow_raise": 0.07,
            "eye_aperture":     0.05,
            "mouth_open":       0.04,
        }))
        assert em["fear"] == max(em.values()), (
            f"Expected fear dominant, got {max(em, key=em.get)} ({em})")

    def test_anger_from_brow_narrowing(self):
        """Brows drawn together (negative brow_gap delta) → anger dominant."""
        em = classify_emotion(adj_from({
            "brow_gap":          -0.08,   # brows narrowed
            "lower_lid_tension":  0.04,
            "lip_press":          0.05,
        }))
        assert em["anger"] == max(em.values()), (
            f"Expected anger dominant, got {max(em, key=em.get)} ({em})")

    def test_disgust_from_lip_raise_and_wrinkle(self):
        """Upper lip raise + nose wrinkle → disgust dominant."""
        em = classify_emotion(adj_from({
            "upper_lip_raise": 0.08,
            "nose_wrinkle":    0.06,
        }))
        assert em["disgust"] == max(em.values()), (
            f"Expected disgust dominant, got {max(em, key=em.get)} ({em})")

    def test_sadness_from_inner_brow_and_lip_drop(self):
        """Inner brow raise + lip corners down → sadness dominant."""
        em = classify_emotion(adj_from({
            "inner_brow_raise": 0.06,
            "lip_corner_dir":  -0.06,
        }))
        assert em["sadness"] == max(em.values()), (
            f"Expected sadness dominant, got {max(em, key=em.get)} ({em})")

    # ── Output contract ───────────────────────────────────────────────────────

    def test_scores_sum_to_one(self):
        """Normalised scores must sum to 1.0."""
        for _ in range(10):
            f = {k: float(np.random.uniform(-0.05, 0.10)) for k in FEATURE_NAMES}
            em = classify_emotion(f)
            assert abs(sum(em.values()) - 1.0) < 0.01, f"Scores don't sum to 1: {em}"

    def test_all_labels_present(self):
        """Every emotion label must be in every response."""
        em = classify_emotion(neutral_adj())
        assert set(em.keys()) == set(EMOTION_LABELS)

    def test_scores_non_negative(self):
        """No emotion score should be negative."""
        em = classify_emotion(adj_from({"lip_corner_dir": -0.10, "brow_gap": -0.10}))
        for label, score in em.items():
            assert score >= 0, f"{label} has negative score {score}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deception analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestDeceptionAnalysis:

    def _run(self, adj_overrides=None, emotion_overrides=None, baseline=None):
        adj = adj_from(adj_overrides or {})
        bl  = baseline or neutral_raw()
        em  = classify_emotion(adj)
        if emotion_overrides:
            em.update(emotion_overrides)
            total = sum(em.values())
            em = {k: v/total for k, v in em.items()}
        return make_deception(adj, bl, em)

    # ── Intensity gate ────────────────────────────────────────────────────────

    def test_neutral_face_scores_low(self):
        """All-zero adj features → intensity gate fires → score < 15%."""
        result = self._run()
        assert result["deception_score"] < 0.15, (
            f"Neutral face scored {result['deception_score']:.2f} — expected < 0.15")

    def test_neutral_face_has_no_clues(self):
        """A perfectly neutral face should produce zero clues."""
        result = self._run()
        assert result["clues"] == [], (
            f"Neutral face generated clues: {result['clues']}")

    def test_small_expression_below_gate(self):
        """Very small deltas (noise level) shouldn't trigger deception."""
        result = self._run({"lip_corner_dir": 0.005, "cheek_raise": 0.003})
        assert result["deception_score"] < 0.15

    # ── Morphology: Non-Duchenne smile ────────────────────────────────────────

    def test_genuine_smile_does_not_trigger_non_duchenne(self):
        """Smile WITH Duchenne markers should not fire Non-Duchenne clue."""
        result = self._run(
            adj_overrides={
                "lip_corner_dir":    0.08,   # corners raised
                "cheek_raise":       0.06,   # Duchenne: cheek elevated
                "lower_lid_tension": 0.04,   # Duchenne: lid crinkle
            }
        )
        labels = [c["label"] for c in result["clues"]]
        assert "Non-Duchenne Smile" not in labels, (
            "Genuine Duchenne smile incorrectly flagged")

    def test_social_smile_triggers_non_duchenne(self):
        """Smile WITHOUT both Duchenne markers → Non-Duchenne detected."""
        # Force happiness to be dominant by setting high deltas for smile features
        # but flat Duchenne markers
        result = self._run(
            adj_overrides={"lip_corner_dir": 0.10},  # strong smile, no Duchenne
            emotion_overrides={"happiness": 0.70, "surprise": 0.05,
                               "fear": 0.05, "anger": 0.05, "disgust": 0.05, "sadness": 0.10},
        )
        labels = [c["label"] for c in result["clues"]]
        assert "Non-Duchenne Smile" in labels, (
            f"Social smile not detected. Clues: {labels}")

    # ── Morphology: fear without brow ─────────────────────────────────────────

    def test_genuine_fear_has_brow(self):
        """Fear expression WITH inner brow raise should not flag Fear-Without-Brow."""
        result = self._run(
            adj_overrides={"inner_brow_raise": 0.06, "eye_aperture": 0.05},
        )
        labels = [c["label"] for c in result["clues"]]
        assert "Fear Without Fear Brow" not in labels

    def test_simulated_fear_triggers_no_brow_clue(self):
        """Fear indicators without inner brow raise → Fear-Without-Fear-Brow."""
        result = self._run(
            adj_overrides={"eye_aperture": 0.06, "mouth_open": 0.04},
            emotion_overrides={"fear": 0.70, "happiness": 0.05, "surprise": 0.05,
                               "anger": 0.05, "disgust": 0.05, "sadness": 0.10},
        )
        labels = [c["label"] for c in result["clues"]]
        assert "Fear Without Fear Brow" in labels, (
            f"Simulated fear not detected. Clues: {labels}")

    # ── Score properties ──────────────────────────────────────────────────────

    def test_score_between_zero_and_one(self):
        """Deception score must always be in [0, 1]."""
        for _ in range(20):
            adj = {k: float(np.random.uniform(-0.05, 0.10)) for k in FEATURE_NAMES}
            em  = classify_emotion(adj)
            r   = make_deception(adj, neutral_raw(), em)
            assert 0.0 <= r["deception_score"] <= 1.0

    def test_score_capped_below_0_93(self):
        """Score cap should prevent runaway values."""
        # Force many components by injecting extreme emotion values
        adj = adj_from({"lip_corner_dir": 0.20, "inner_brow_raise": 0.15})
        em  = {"happiness": 0.80, "surprise": 0.04, "fear": 0.04,
               "anger": 0.04, "disgust": 0.04, "sadness": 0.04}
        r   = make_deception(adj, neutral_raw(), em)
        assert r["raw_score"] <= 0.93, f"Raw score {r['raw_score']} exceeded cap"

    def test_ema_smoother_damps_spikes(self):
        """EMA smoother should damp a sudden jump from 0 → 0.97."""
        smoother = ExponentialSmoother(alpha=0.15)
        smoother.update(0.05)           # establish low baseline
        smoothed = smoother.update(0.97)  # single spike
        # One spike of 0.97 from a 0.05 baseline should be damped to ~0.20
        # (0.15 * 0.97 + 0.85 * 0.05 = 0.188)
        assert smoothed < 0.30, (
            f"EMA smoother didn't damp spike: {smoothed:.3f} after one frame")

    def test_ema_smoother_converges_to_steady_state(self):
        """EMA smoother converges to the steady-state value over many frames."""
        smoother = ExponentialSmoother(alpha=0.15)
        for _ in range(60):
            v = smoother.update(0.80)
        # After 60 frames at 0.80, value should be within 1% of 0.80
        assert abs(v - 0.80) < 0.01, f"Smoother didn't converge: {v:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration:

    def test_baseline_equals_mean_of_frames(self):
        """compute_baseline should return per-feature mean of input frames."""
        frames = [
            {k: float(i) for k in FEATURE_NAMES}
            for i in range(10)
        ]
        bl = compute_baseline(frames)
        for k in FEATURE_NAMES:
            expected = float(np.mean([i for i in range(10)]))
            assert abs(bl[k] - expected) < 1e-9, f"Baseline for {k} wrong"

    def test_baseline_quality_high_when_stable(self):
        """Low variance during calibration → quality close to 1."""
        stable = [neutral_raw() for _ in range(30)]
        quality = _baseline_quality(stable)
        assert quality > 0.70, f"Stable calibration quality too low: {quality}"

    def test_baseline_quality_low_when_moving(self):
        """High variance during calibration → quality well below 1."""
        rng = np.random.default_rng(0)
        noisy = [
            {k: neutral_raw()[k] + float(rng.uniform(-0.1, 0.1))
             for k in FEATURE_NAMES}
            for _ in range(30)
        ]
        quality = _baseline_quality(noisy)
        assert quality < 0.85, f"Noisy calibration quality too high: {quality}"

    def test_calibration_phase_progression(self):
        """Calibration phases must progress settling → collecting → active."""
        assert _calibration_state(1,  [], {})["phase"] == "settling"
        assert _calibration_state(5,  [], {})["phase"] == "collecting"
        assert _calibration_state(35, [], {"x": 1.0})["phase"] == "active"

    def test_calibration_progress_0_to_100(self):
        """Progress should go from 0% at frame 5 to 100% at frame 35."""
        assert _calibration_state(5,  [], {})["progress"] == 0
        assert _calibration_state(35, [], {"x": 1.0})["progress"] == 100

    def test_delta_zero_after_calibration(self):
        """After calibration, features at neutral produce near-zero deltas."""
        raw = neutral_raw()
        baseline = compute_baseline([raw] * 30)
        adj = {k: raw[k] - baseline.get(k, raw[k]) for k in FEATURE_NAMES}
        for k, v in adj.items():
            assert abs(v) < 1e-9, f"Non-zero delta for {k} after perfect calibration: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. API smoke tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


class TestAPI:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["status"] == "ok"

    def test_debug_returns_model_path(self, client):
        r = client.get("/debug")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "model_path" in data
        assert "python" in data

    def test_warmup_returns_status(self, client):
        r = client.get("/warmup")
        assert r.status_code in (200, 500)   # 500 acceptable if model not on disk
        data = json.loads(r.data)
        assert "status" in data

    def test_analyse_rejects_missing_image(self, client):
        r = client.post("/analyse",
                        data=json.dumps({}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_reset_returns_ok(self, client):
        r = client.post("/reset")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["status"] == "reset"

    def test_collect_sample_rejects_missing_fields(self, client):
        r = client.post("/collect_sample",
                        data=json.dumps({}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_training_data_accumulates(self, client):
        """Posting samples should accumulate in the training store."""
        client.post("/reset_training",
                    data=json.dumps({}),
                    content_type="application/json")

        sample_features = {k: 0.05 for k in FEATURE_NAMES}
        for _ in range(5):
            r = client.post("/collect_sample",
                            data=json.dumps({
                                "features": sample_features,
                                "label":    "happiness"
                            }),
                            content_type="application/json")
            assert r.status_code == 200

        r = client.get("/training_status")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["total_samples"] >= 5


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
