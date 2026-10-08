"""
train_model.py
--------------
Synthetic dataset generated for educational AI/ML demonstration.

Trains two models on the synthetic agricultural dataset:
  1. Crop Recommendation  — multi-class classifier (RandomForest)
  2. Yield Estimation     — regression model (GradientBoosting)

Usage:
    python generate_dataset.py   # generate data first
    python train_model.py        # then train

Outputs:
    models/crop_recommender.pkl
    models/yield_estimator.pkl
    models/label_encoder.pkl
    models/feature_scaler.pkl
    reports/training_report.txt
"""

import os
import sys
import pickle
import random

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

DATA_CSV   = os.path.join("data", "synthetic_crop_data.csv")
MODEL_DIR  = "models"
REPORT_DIR = "reports"

NOTE = "Synthetic dataset generated for educational AI/ML demonstration."

# ── guard: data must exist ────────────────────────────────────────────────────
if not os.path.exists(DATA_CSV):
    print(f"[ERROR] Dataset not found: {DATA_CSV}")
    print("  Run:  python generate_dataset.py")
    sys.exit(1)

# ── imports (after guard so missing deps give a cleaner message) ──────────────
try:
    import csv
    import math
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, classification_report,
        mean_absolute_error, mean_squared_error, r2_score,
    )
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Run:  pip install -r requirements.txt")
    sys.exit(1)

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("AI/ML Crop Advisory System — Model Training")
print(NOTE)
print("=" * 60)
print(f"\n[1/6] Loading data from {DATA_CSV} ...")

FEATURE_COLS = [
    "nitrogen_N", "phosphorus_P", "potassium_K",
    "temperature_C", "humidity_pct", "soil_pH", "rainfall_mm",
]
TARGET_CLASS = "crop_type"
TARGET_REGR  = "estimated_yield"

rows = []
with open(DATA_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

X_raw   = [[float(r[c]) for c in FEATURE_COLS] for r in rows]
y_class = [r[TARGET_CLASS]  for r in rows]
y_regr  = [float(r[TARGET_REGR]) for r in rows]

X = np.array(X_raw)
y_regr = np.array(y_regr)

print(f"   Records loaded  : {len(rows)}")
print(f"   Feature columns : {FEATURE_COLS}")
print(f"   Unique crops    : {len(set(y_class))}")

# ── encode labels ─────────────────────────────────────────────────────────────
print("\n[2/6] Encoding labels ...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)
print(f"   Classes: {list(le.classes_)}")

# ── scale features ────────────────────────────────────────────────────────────
print("\n[3/6] Scaling features ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── train / test split ────────────────────────────────────────────────────────
print("\n[4/6] Splitting into train / test sets (80 / 20) ...")
(X_train, X_test,
 yc_train, yc_test,
 yr_train, yr_test) = train_test_split(
    X_scaled, y_encoded, y_regr,
    test_size=0.20, random_state=SEED, stratify=y_encoded,
)
print(f"   Training samples : {len(X_train)}")
print(f"   Test samples     : {len(X_test)}")

# ── model 1: crop recommender (classifier) ────────────────────────────────────
print("\n[5/6] Training models ...")
print("   -> Crop Recommender (RandomForestClassifier) ...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=4,
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(X_train, yc_train)
yc_pred = clf.predict(X_test)
clf_acc = accuracy_score(yc_test, yc_pred)
cv_scores = cross_val_score(clf, X_scaled, y_encoded, cv=5,
                            scoring="accuracy", n_jobs=-1)
print(f"      Test accuracy  : {clf_acc:.4f}")
print(f"      CV accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── model 2: yield estimator (regressor) ─────────────────────────────────────
print("   -> Yield Estimator (GradientBoostingRegressor) ...")
reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    random_state=SEED,
)
reg.fit(X_train, yr_train)
yr_pred = reg.predict(X_test)
mae  = mean_absolute_error(yr_test, yr_pred)
rmse = math.sqrt(mean_squared_error(yr_test, yr_pred))
r2   = r2_score(yr_test, yr_pred)
print(f"      MAE  : {mae:.4f}")
print(f"      RMSE : {rmse:.4f}")
print(f"      R²   : {r2:.4f}")

# ── save artefacts ────────────────────────────────────────────────────────────
print("\n[6/6] Saving model artefacts ...")

def _save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"   Saved: {path}")

_save(clf,    os.path.join(MODEL_DIR, "crop_recommender.pkl"))
_save(reg,    os.path.join(MODEL_DIR, "yield_estimator.pkl"))
_save(le,     os.path.join(MODEL_DIR, "label_encoder.pkl"))
_save(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))

# ── training report ───────────────────────────────────────────────────────────
report_path = os.path.join(REPORT_DIR, "training_report.txt")
clf_report  = classification_report(
    yc_test, yc_pred, target_names=le.classes_
)

feat_importance = sorted(
    zip(FEATURE_COLS, clf.feature_importances_),
    key=lambda x: x[1], reverse=True,
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write("AI/ML CROP ADVISORY SYSTEM — TRAINING REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"{NOTE}\n\n")
    f.write("DATASET\n")
    f.write(f"  Source  : {DATA_CSV}\n")
    f.write(f"  Records : {len(rows)}\n")
    f.write(f"  Features: {', '.join(FEATURE_COLS)}\n\n")
    f.write("CROP RECOMMENDER — RandomForestClassifier\n")
    f.write(f"  Test accuracy       : {clf_acc:.4f}\n")
    f.write(f"  5-fold CV accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("  Feature importances (classifier):\n")
    for feat, imp in feat_importance:
        bar = "█" * int(imp * 40)
        f.write(f"    {feat:<18} {imp:.4f}  {bar}\n")
    f.write("\n  Per-class report:\n")
    f.write(clf_report + "\n")
    f.write("YIELD ESTIMATOR — GradientBoostingRegressor\n")
    f.write(f"  MAE  : {mae:.4f}\n")
    f.write(f"  RMSE : {rmse:.4f}\n")
    f.write(f"  R²   : {r2:.4f}\n\n")
    f.write("ARTEFACTS\n")
    f.write(f"  models/crop_recommender.pkl\n")
    f.write(f"  models/yield_estimator.pkl\n")
    f.write(f"  models/label_encoder.pkl\n")
    f.write(f"  models/feature_scaler.pkl\n")

print(f"   Report : {report_path}")
print("\n" + "=" * 60)
print("Training complete.")
print(f"  Classifier accuracy : {clf_acc:.4f}")
print(f"  Yield estimator R²  : {r2:.4f}")
print("Run  python predict.py  to make predictions.")
print("=" * 60)
