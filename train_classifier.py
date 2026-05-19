"""
VERITY — Automated Classifier Training Pipeline
================================================
Trains a scikit-learn emotion classifier on real facial expression datasets,
replacing VERITY's rule-based classify_emotion() with a data-driven model.

Supported dataset formats:
  1. FER2013 folder  — train/test subfolders, one folder per emotion  ← DEFAULT
  2. FER2013 CSV     — legacy single-file format
  3. RAF-DB          — folder-per-class structure (Basic split)
  4. CK+             — AUs + emotion labels, 7-class split

Usage
-----
# FER2013 folder format (most common Kaggle download)
python train_classifier.py --dataset fer2013_folder --data_dir fer2013/

# FER2013 CSV format
python train_classifier.py --dataset fer2013 --csv fer2013.csv

# RAF-DB
python train_classifier.py --dataset rafdb --data_dir RAF-DB/basic/Image/aligned

# CK+
python train_classifier.py --dataset ck+ --data_dir CK+/cohn-kanade-images --emotion_dir CK+/Emotion

# Combine datasets
python train_classifier.py --dataset fer2013_folder rafdb --data_dir fer2013/

Output
------
models/emotion_classifier.pkl  — sklearn pipeline (scaler + classifier)
models/label_encoder.pkl       — LabelEncoder for emotion names
models/feature_order.json      — canonical feature vector order
models/training_report.json    — accuracy, confusion matrix, per-class F1
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **_):
            self._it = iter(iterable) if iterable is not None else None
            self.n = 0
            self._total = total
            self._desc = desc
        def __iter__(self): return self
        def __next__(self):
            item = next(self._it)
            self.n += 1
            if self.n % 200 == 0:
                print(f"  {self._desc}: {self.n}/{self._total or '?'}")
            return item
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def set_postfix(self, **_): pass

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Emotion label mappings
# ──────────────────────────────────────────────────────────
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# Folder name → VERITY label (covers both FER2013 folder and RAF-DB variants)
FOLDER_LABEL_MAP = {
    # FER2013 folder names
    "angry":     "anger",
    "anger":     "anger",
    "disgust":   "disgust",
    "fear":      "fear",
    "fearful":   "fear",
    "happy":     "happiness",
    "happiness": "happiness",
    "neutral":   "neutral",
    "sad":       "sadness",
    "sadness":   "sadness",
    "surprise":  "surprise",
    # RAF-DB numbered folders
    "1": "surprise", "2": "fear", "3": "disgust",
    "4": "happiness", "5": "sadness", "6": "anger", "7": "neutral",
}

FER_CSV_MAP = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness",
               4: "sadness", 5: "surprise", 6: "neutral"}

CKP_MAP = {1: "anger", 2: "disgust", 3: "fear", 4: "happiness",
           5: "sadness", 6: "surprise", 7: "neutral"}

# ──────────────────────────────────────────────────────────
# Canonical feature vector order
# Must stay identical to VERITY's extract_features() output
# ──────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "inner_brow_raise", "outer_brow_raise", "brow_gap",
    "eye_aperture", "lower_lid_tension", "cheek_raise",
    "mouth_width", "mouth_open", "lip_corner_dir",
    "upper_lip_raise", "lip_press", "nose_wrinkle", "jaw_drop",
]

# ──────────────────────────────────────────────────────────
# MediaPipe face mesh
# ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
_face_mesh   = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.40,
    min_tracking_confidence=0.40,
    static_image_mode=True,
)

LANDMARKS = {
    "inner_brow_left":   [107, 66, 105, 63, 70],
    "inner_brow_right":  [336, 296, 334, 293, 300],
    "outer_brow_left":   [46, 53, 52, 65, 55],
    "outer_brow_right":  [285, 295, 282, 283, 276],
    "upper_lid_left":    [159, 158, 157, 173, 133],
    "upper_lid_right":   [386, 385, 384, 398, 362],
    "lower_lid_left":    [145, 144, 163, 7],
    "lower_lid_right":   [374, 373, 390, 249],
    "cheek_left":        [116, 117, 118, 119, 120],
    "cheek_right":       [345, 346, 347, 348, 349],
    "mouth_corner_left": [61, 146, 91, 181, 84],
    "mouth_corner_right":[291, 375, 321, 405, 314],
    "upper_lip":         [13, 312, 311, 310, 415, 308, 78, 191, 80, 81, 82],
    "lower_lip":         [14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13],
    "jaw":               [152, 377, 400, 378, 379, 365, 397, 288],
    "nose_bridge":       [6, 197, 195, 5],
    "nose_lower":        [4, 240, 98, 97, 2, 326, 327, 460],
}

# ──────────────────────────────────────────────────────────
# Feature extraction (identical to app.py)
# ──────────────────────────────────────────────────────────

def _lm(lms, idx, w, h):
    p = lms[idx]
    return np.array([p.x * w, p.y * h])

def _spread(lms, indices, w, h):
    pts = np.array([_lm(lms, i, w, h) for i in indices])
    c   = pts.mean(axis=0)
    return float(np.mean(np.linalg.norm(pts - c, axis=1)))

def _lift(lms, tip, base, w, h):
    return float(_lm(lms, base, w, h)[1] - _lm(lms, tip, w, h)[1])

def _eu(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def extract_features(lms, w, h):
    lo  = _lm(lms, 33,  w, h)
    ro  = _lm(lms, 263, w, h)
    iod = max(_eu(lo, ro), 1.0)
    n   = lambda d: d / iod

    inner_brow_raise = n((_lift(lms, 107, 33, w, h) + _lift(lms, 336, 263, w, h)) / 2)
    brow_gap         = n(_eu(_lm(lms, 107, w, h), _lm(lms, 336, w, h)))
    outer_brow_raise = n((_lift(lms, 70, 33, w, h) + _lift(lms, 300, 263, w, h)) / 2)

    et_L = _lm(lms, 159, w, h); eb_L = _lm(lms, 145, w, h)
    et_R = _lm(lms, 386, w, h); eb_R = _lm(lms, 374, w, h)
    eye_aperture = (n(_eu(et_L, eb_L)) + n(_eu(et_R, eb_R))) / 2

    lower_lid_tension = n((_spread(lms, LANDMARKS["lower_lid_left"],  w, h) +
                           _spread(lms, LANDMARKS["lower_lid_right"], w, h)) / 2)

    cheek_raise = n((_lift(lms, 116, 145, w, h) + _lift(lms, 345, 374, w, h)) / 2)

    ml = _lm(lms, 61,  w, h); mr = _lm(lms, 291, w, h)
    mouth_width = n(_eu(ml, mr))

    mt = _lm(lms, 13, w, h); mb = _lm(lms, 14, w, h)
    mouth_open = n(_eu(mt, mb))

    nose_tip       = _lm(lms, 4,  w, h)
    lip_corner_dir = (n(float(nose_tip[1] - ml[1])) + n(float(nose_tip[1] - mr[1]))) / 2

    nose_base       = _lm(lms, 2,  w, h)
    upper_lip_raise = n(float(nose_base[1] - _lm(lms, 13, w, h)[1]))

    lip_press    = 1.0 - min(mouth_open * 10, 1.0)
    nose_wrinkle = n(_spread(lms, LANDMARKS["nose_lower"], w, h))
    jaw_drop     = n(float(_lm(lms, 152, w, h)[1] - mb[1]))

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


def image_to_features(img_bgr, target_size=224):
    h, w = img_bgr.shape[:2]
    if max(h, w) < target_size:
        scale   = target_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        h, w = img_bgr.shape[:2]

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = _face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    return extract_features(res.multi_face_landmarks[0].landmark, w, h)


def features_to_vector(feat_dict):
    return np.array([feat_dict[k] for k in FEATURE_ORDER], dtype=np.float32)

# ──────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────

def load_fer2013_folder(data_dir, splits=("train", "test"), max_per_class=None):
    """
    Load FER2013 in folder format:
        <data_dir>/
            train/
                angry/  happy/  sad/  fear/  disgust/  surprise/  neutral/
            test/
                angry/  ...

    Also works if data_dir itself contains the emotion folders directly.
    """
    data_dir = Path(data_dir)
    samples  = []

    # Auto-detect layout
    subdirs      = [d for d in data_dir.iterdir() if d.is_dir()]
    subdir_names = {d.name.lower() for d in subdirs}
    split_names  = {"train", "test", "valid", "validation", "val"}
    has_split_level = bool(subdir_names & split_names)

    if has_split_level:
        search_dirs = [data_dir / s for s in splits if (data_dir / s).exists()]
        if not search_dirs:
            search_dirs = [d for d in subdirs if d.name.lower() in split_names]
    else:
        search_dirs = [data_dir]

    print(f"\n[FER2013-folder] Root: {data_dir}")
    print(f"  Splits found: {[d.name for d in search_dirs]}")

    class_counts = Counter()
    for search_dir in search_dirs:
        for cls_dir in sorted(search_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = FOLDER_LABEL_MAP.get(cls_dir.name.lower())
            if label is None:
                print(f"  ! Unrecognised folder '{cls_dir.name}' — skipping")
                continue
            imgs = (list(cls_dir.glob("*.jpg"))  + list(cls_dir.glob("*.JPG"))  +
                    list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.JPEG")) +
                    list(cls_dir.glob("*.png"))  + list(cls_dir.glob("*.PNG")))
            for img_path in imgs:
                img = cv2.imread(str(img_path))
                if img is not None:
                    samples.append((img, label))
                    class_counts[label] += 1

    print(f"  Loaded {len(samples)} images:")
    for cls in sorted(class_counts):
        print(f"    {cls:<12} {class_counts[cls]:>5}")

    if max_per_class:
        samples = _cap_per_class(samples, max_per_class)
        print(f"  After cap ({max_per_class}/class): {len(samples)}")

    return samples


def load_fer2013_csv(csv_path, split="Training", max_per_class=None):
    print(f"\n[FER2013-CSV] {csv_path} (split={split}) ...")
    import csv
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Usage", split) != split:
                continue
            label  = FER_CSV_MAP.get(int(row["emotion"]))
            if label is None:
                continue
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            bgr    = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
            samples.append((bgr, label))
    if max_per_class:
        samples = _cap_per_class(samples, max_per_class)
    print(f"  {len(samples)} samples")
    return samples


def load_rafdb(data_dir, split="train"):
    print(f"\n[RAF-DB] {data_dir} (split={split}) ...")
    data_dir = Path(data_dir)
    samples  = []
    for cls_dir in sorted(data_dir.glob(f"{split}/*")):
        if not cls_dir.is_dir():
            continue
        label = FOLDER_LABEL_MAP.get(cls_dir.name.lower())
        if label is None:
            continue
        for img_path in cls_dir.glob("*.[jJpP][pPnN]*"):
            img = cv2.imread(str(img_path))
            if img is not None:
                samples.append((img, label))
    print(f"  {len(samples)} samples")
    return samples


def load_ckplus(image_dir, emotion_dir):
    print(f"\n[CK+] {image_dir} ...")
    image_dir   = Path(image_dir)
    emotion_dir = Path(emotion_dir)
    samples     = []
    for emo_file in sorted(emotion_dir.rglob("*_emotion.txt")):
        try:
            emotion_int = int(float(emo_file.read_text().strip()))
        except Exception:
            continue
        label = CKP_MAP.get(emotion_int)
        if label is None:
            continue
        parts      = emo_file.parts
        frames_dir = image_dir / parts[-3] / parts[-2]
        frames     = sorted(frames_dir.glob("*.png"))
        if not frames:
            continue
        img = cv2.imread(str(frames[-1]))
        if img is not None:
            samples.append((img, label))
    print(f"  {len(samples)} samples")
    return samples


def _cap_per_class(samples, max_n):
    import random
    from collections import defaultdict
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    result = []
    for cls, items in by_class.items():
        random.shuffle(items)
        result.extend(items[:max_n])
    return result

# ──────────────────────────────────────────────────────────
# Feature extraction pass
# ──────────────────────────────────────────────────────────

def extract_all(samples, desc="Extracting"):
    X, y    = [], []
    skipped = 0
    bar     = tqdm(samples, total=len(samples), desc=desc)
    for img_bgr, label in bar:
        feats = image_to_features(img_bgr)
        if feats is None:
            skipped += 1
            bar.set_postfix(skipped=skipped)
            continue
        X.append(features_to_vector(feats))
        y.append(label)
    print(f"\n  Extracted {len(X)} / {len(samples)}  ({skipped} skipped — no face detected)")
    return np.array(X, dtype=np.float32), np.array(y)

# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────

def train(X, y, output_dir="models"):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix

    os.makedirs(output_dir, exist_ok=True)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n[Training] Class distribution:")
    counts = Counter(y)
    max_c  = max(counts.values())
    for cls in sorted(counts):
        bar = "█" * (counts[cls] * 40 // max_c)
        print(f"  {cls:<12} {counts[cls]:>5}  {bar}")

    candidates = {
        "MLP (128-64)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                random_state=42, learning_rate_init=0.001,
            )),
        ]),
        "RandomForest 300": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42, subsample=0.8,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_score, best_pipeline = None, -1, None

    for name, pipeline in candidates.items():
        print(f"\n  Evaluating: {name}")
        t0     = time.time()
        scores = cross_val_score(pipeline, X, y_enc, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        print(f"    F1-macro: {scores.mean():.3f} ± {scores.std():.3f}  "
              f"({time.time()-t0:.1f}s)")
        if scores.mean() > best_score:
            best_score, best_name, best_pipeline = scores.mean(), name, pipeline

    print(f"\n[Training] Winner: {best_name}  (F1={best_score:.3f})")
    print("[Training] Fitting final model on full dataset ...")
    best_pipeline.fit(X, y_enc)

    y_pred = best_pipeline.predict(X)
    report = classification_report(y_enc, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print("\n" + classification_report(y_enc, y_pred, target_names=le.classes_))

    with open(f"{output_dir}/emotion_classifier.pkl", "wb") as f:
        pickle.dump(best_pipeline, f, protocol=4)
    with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f, protocol=4)
    with open(f"{output_dir}/feature_order.json", "w") as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    with open(f"{output_dir}/training_report.json", "w") as f:
        json.dump({"model": best_name, "cv_f1_macro": round(best_score, 4),
                   "classes": list(le.classes_),
                   "classification_report": report,
                   "confusion_matrix": confusion_matrix(y_enc, y_pred).tolist(),
                   "n_train": int(len(X))}, f, indent=2)

    print(f"✓ Saved to {output_dir}/")
    return best_pipeline, le

# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VERITY emotion classifier trainer")
    parser.add_argument("--dataset", nargs="+", default=["fer2013_folder"],
                        choices=["fer2013_folder", "fer2013", "rafdb", "ck+"])
    parser.add_argument("--data_dir",    default="fer2013",
                        help="Root folder (fer2013/, RAF-DB/, or CK+ images)")
    parser.add_argument("--csv",         default="fer2013.csv",
                        help="FER2013 CSV path (only for --dataset fer2013)")
    parser.add_argument("--emotion_dir", default=None,
                        help="CK+ Emotion label directory")
    parser.add_argument("--splits",      nargs="+", default=["train", "test"],
                        help="Which split folders to use (default: train test)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap per class — useful for a quick test run")
    parser.add_argument("--output_dir",  default="models")
    parser.add_argument("--validate",    action="store_true",
                        help="Run validation on test split after training")
    args = parser.parse_args()

    all_train, all_val = [], []

    for ds in args.dataset:
        if ds == "fer2013_folder":
            if not os.path.exists(args.data_dir):
                print(f"✗ Folder not found: {args.data_dir}")
                print("  Use --data_dir to point to your fer2013 folder")
                sys.exit(1)
            train_splits = [s for s in args.splits if s != "test"]
            val_splits   = ["test"] if "test" in args.splits else []
            all_train.extend(load_fer2013_folder(
                args.data_dir,
                splits=train_splits or args.splits,
                max_per_class=args.max_per_class))
            if val_splits:
                all_val.extend(load_fer2013_folder(
                    args.data_dir, splits=val_splits))

        elif ds == "fer2013":
            if not os.path.exists(args.csv):
                print(f"✗ CSV not found: {args.csv}")
                sys.exit(1)
            all_train.extend(load_fer2013_csv(
                args.csv, split="Training", max_per_class=args.max_per_class))
            all_val.extend(load_fer2013_csv(args.csv, split="PublicTest"))

        elif ds == "rafdb":
            all_train.extend(load_rafdb(args.data_dir, split="train"))
            all_val.extend(load_rafdb(args.data_dir, split="test"))

        elif ds == "ck+":
            if not args.emotion_dir:
                print("✗ --emotion_dir required for CK+")
                sys.exit(1)
            ck = load_ckplus(args.data_dir, args.emotion_dir)
            split = int(len(ck) * 0.8)
            all_train.extend(ck[:split])
            all_val.extend(ck[split:])

    if not all_train:
        print("✗ No training samples loaded — check your paths.")
        sys.exit(1)

    print(f"\n{'='*56}")
    print(f"  Training samples   : {len(all_train)}")
    print(f"  Validation samples : {len(all_val)}")
    print(f"{'='*56}")

    print("\n[Step 1/2] Extracting MediaPipe features ...")
    print("  FER2013 full set takes ~15–40 min on CPU")
    X_train, y_train = extract_all(all_train, desc="Train")

    if len(X_train) < 50:
        print("✗ Too few usable samples — check image paths.")
        sys.exit(1)

    print("\n[Step 2/2] Training classifiers ...")
    pipeline, le = train(X_train, y_train, output_dir=args.output_dir)

    if args.validate and all_val:
        print("\n[Validation]")
        X_val, y_val = extract_all(all_val, desc="Val")
        from sklearn.metrics import classification_report, accuracy_score
        y_enc  = le.transform(y_val)
        y_pred = pipeline.predict(X_val)
        print(f"Accuracy: {accuracy_score(y_enc, y_pred):.3f}")
        print(classification_report(y_enc, y_pred, target_names=le.classes_))

    print("\n✓ Done. Commit the models/ folder to git and redeploy on Render.")


if __name__ == "__main__":
    main()
