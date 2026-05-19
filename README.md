# VERITY — Deception Analysis System
### Based on Paul Ekman & Wallace V. Friesen, *Unmasking the Face* (2003)

A real-time facial deception detection app that applies Ekman's four-factor framework:
**Morphology · Timing · Micro-Expressions · Leakage**

---

## Requirements

- macOS (10.15+) or Linux
- Python 3.9+
- Webcam
- Internet connection (first run — downloads MediaPipe models)

---

## Quick Start

```bash
# 1. Make launcher executable
chmod +x run.sh

# 2. Launch (installs deps automatically)
./run.sh
```

The browser opens automatically at **http://localhost:5050**

---

## Manual Setup (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
# Open http://localhost:5050
```

---

## How It Works

### Ekman's Framework (Ch. 11 — "Facial Deceit")

| Factor | Description | Implementation |
|--------|-------------|----------------|
| **Morphology** | Upper vs lower face incongruence | MediaPipe 468-point mesh; 13 scalar muscle features |
| **Timing** | Onset speed, duration, offset anomalies | 90-frame rolling buffer (~3s at 30fps) |
| **Micro-expressions** | Brief 1–4 frame suppressed emotion flashes | Frame-by-frame delta detection |
| **Leakage** | True emotion bleeding through managed expression | Cross-region incongruence analysis |

### Key Ekman Principles Applied

- **Lower face = primary management zone** — mouth/lips most controlled; deception often managed here
- **Upper face = primary leakage zone** — brow/forehead hardest to fake; fear and sad brow especially reliable
- **Duchenne smile detection** — genuine happiness requires cheek raise + lower lid tension (orbicularis oculi); absent in social/fake smiles
- **Fear brow reliability** — inner brow raise is not an emblem/punctuator; its presence indicates genuine fear
- **Sad brow reliability** — inner brow elevation rare as a punctuator; absent in simulated sadness
- **Prolonged surprise = deception clue** — surprise is always brief; duration > ~0.7s indicates simulation

### Deception Score Interpretation

| Score | Verdict | Meaning |
|-------|---------|---------|
| 0–25% | TRUTHFUL | No significant deception indicators detected |
| 25–55% | INCONCLUSIVE | Some indicators present; insufficient evidence |
| 55–100% | DECEPTION LIKELY | Multiple simultaneous Ekman deception clues |

---

## Facial Regions Monitored

```
BROW/FOREHEAD  ← Primary leakage zone (Ekman p.146)
  • Inner brow raise (fear/sadness — hard to fake)
  • Outer brow raise (surprise emblem — easy to fake)
  • Brow convergence (anger/concentration emblem)

EYES / EYELIDS ← Secondary leakage zone
  • Eye aperture (fear stare; surprise)
  • Lower lid tension (Duchenne marker; anger)
  • Cheek elevation (Duchenne happiness)

MOUTH / LIPS   ← Primary management zone (Ekman p.146)
  • Lip corner direction (happiness vs sadness)
  • Upper lip raise (disgust)
  • Lip press (anger)
  • Mouth width/aperture (fear/surprise)

NOSE           ← Disgust marker
  • Nose wrinkle
```

---

## Architecture

```
Browser (Webcam) → Flask /analyse → MediaPipe Face Mesh
                                  → Feature Extraction (13 features)
                                  → Emotion Classification (rule-based)
                                  → Ekman Deception Analysis
                                  → JSON response → UI update
```

**Stack**: Python 3 · Flask · MediaPipe · OpenCV · Vanilla JS

---

## Ethical Notice

This tool is built for **educational and research purposes** to understand Ekman's facial expression science. No system can reliably detect deception with certainty — Ekman's own research notes individual variation, cultural context, and baseline knowledge as essential factors. Do not use this as evidence in any legal, employment, or consequential decision-making context.

---

*Reference: Paul Ekman & Wallace V. Friesen, "Unmasking the Face: A Guide to Recognizing Emotions From Facial Expressions" (ISHK, 2003)*
