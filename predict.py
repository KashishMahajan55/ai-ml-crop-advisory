"""
predict.py
----------
Synthetic dataset generated for educational AI/ML demonstration.

Loads trained models and predicts:
  1. Recommended crop — given soil & climate parameters
  2. Estimated yield  — for the recommended (or specified) crop

Usage (interactive):
    python predict.py

Usage (batch from CSV):
    python predict.py --batch data/synthetic_crop_data.csv --out results/predictions.csv

All predictions are demonstrations on synthetic data only.
"""

import os
import sys
import pickle
import argparse
import csv

NOTE = "Synthetic dataset generated for educational AI/ML demonstration."

MODEL_DIR = "models"
RESULTS_DIR = "results"

FEATURE_COLS = [
    "nitrogen_N", "phosphorus_P", "potassium_K",
    "temperature_C", "humidity_pct", "soil_pH", "rainfall_mm",
]

FEATURE_LABELS = {
    "nitrogen_N":    ("Nitrogen (N)",       "kg/ha",  0,   200),
    "phosphorus_P":  ("Phosphorus (P)",      "kg/ha",  0,   120),
    "potassium_K":   ("Potassium (K)",       "kg/ha",  0,   150),
    "temperature_C": ("Temperature",         "°C",     0,    50),
    "humidity_pct":  ("Humidity",            "%",      0,   100),
    "soil_pH":       ("Soil pH",             "",       3.0,  10.0),
    "rainfall_mm":   ("Rainfall",            "mm/mo",  0,   400),
}

# ── load artefacts ─────────────────────────────────────────────────────────────

def _load(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] Model artefact not found: {path}")
        print("  Run:  python train_model.py")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models():
    clf    = _load("crop_recommender.pkl")
    reg    = _load("yield_estimator.pkl")
    le     = _load("label_encoder.pkl")
    scaler = _load("feature_scaler.pkl")
    return clf, reg, le, scaler


# ── prediction helpers ─────────────────────────────────────────────────────────

def predict_single(features, clf, reg, le, scaler):
    """
    features : list of 7 floats in FEATURE_COLS order
    Returns  : (top3_crops_with_prob, predicted_yield)
    """
    import numpy as np
    X = np.array([features])
    X_s = scaler.transform(X)

    # top-3 crop probabilities
    proba = clf.predict_proba(X_s)[0]
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [(le.classes_[i], round(float(proba[i]) * 100, 1)) for i in top3_idx]

    # yield estimate
    yield_est = round(float(reg.predict(X_s)[0]), 3)
    return top3, yield_est


# ── interactive mode ──────────────────────────────────────────────────────────

def interactive_mode(clf, reg, le, scaler):
    print("=" * 60)
    print("AI/ML Crop Advisory System — Interactive Predictor")
    print(NOTE)
    print("=" * 60)
    print("Enter soil and climate parameters below.")
    print("(Press Ctrl+C to exit)\n")

    while True:
        try:
            features = []
            for col in FEATURE_COLS:
                label, unit, lo, hi = FEATURE_LABELS[col]
                prompt = f"  {label}"
                if unit:
                    prompt += f" ({unit})"
                prompt += f" [{lo}–{hi}]: "
                while True:
                    raw = input(prompt).strip()
                    try:
                        val = float(raw)
                        if not (lo <= val <= hi):
                            print(f"    [!] Expected value between {lo} and {hi}. Try again.")
                            continue
                        features.append(val)
                        break
                    except ValueError:
                        print("    [!] Please enter a numeric value.")

            top3, yield_est = predict_single(features, clf, reg, le, scaler)

            print("\n" + "-" * 40)
            print("RESULTS  (synthetic demonstration only)")
            print("-" * 40)
            print("  Recommended crops:")
            for rank, (crop, prob) in enumerate(top3, 1):
                bar = "#" * int(prob / 2.5)
                print(f"    {rank}. {crop:<14} {prob:5.1f}%  {bar}")
            print(f"\n  Estimated yield for top recommendation:")
            print(f"    {top3[0][0]}: {yield_est} tonnes/hectare")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n[Exit]")
            break


# ── batch mode ─────────────────────────────────────────────────────────────────

def batch_mode(input_csv, output_csv, clf, reg, le, scaler):
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    total = len(rows)
    print(f"Running batch predictions on {total} records...")

    out_fields = (
        ["record_id"] +
        FEATURE_COLS +
        ["top1_crop", "top1_prob_pct", "top2_crop", "top2_prob_pct",
         "top3_crop", "top3_prob_pct", "estimated_yield_t_ha", "dataset_note"]
    )

    results = []
    for idx, row in enumerate(rows, 1):
        features = [float(row[c]) for c in FEATURE_COLS]
        top3, yield_est = predict_single(features, clf, reg, le, scaler)
        if idx % 500 == 0 or idx == total:
            print(f"  [{idx}/{total}] processed...", flush=True)
        results.append({
            "record_id":             row.get("record_id", ""),
            **{c: row[c] for c in FEATURE_COLS},
            "top1_crop":             top3[0][0],
            "top1_prob_pct":         top3[0][1],
            "top2_crop":             top3[1][0],
            "top2_prob_pct":         top3[1][1],
            "top3_crop":             top3[2][0],
            "top3_prob_pct":         top3[2][1],
            "estimated_yield_t_ha":  yield_est,
            "dataset_note":          NOTE,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"[OK] Batch predictions written: {output_csv}  ({len(results)} records)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI/ML Crop Advisory — predict crop and yield from soil/climate data.\n"
                    f"({NOTE})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", metavar="INPUT_CSV",
                        help="Run batch predictions on an existing CSV file.")
    parser.add_argument("--out",   metavar="OUTPUT_CSV",
                        default=os.path.join(RESULTS_DIR, "predictions.csv"),
                        help="Output CSV path for batch mode (default: results/predictions.csv)")
    args = parser.parse_args()

    clf, reg, le, scaler = load_models()

    if args.batch:
        if not os.path.exists(args.batch):
            print(f"[ERROR] File not found: {args.batch}")
            sys.exit(1)
        batch_mode(args.batch, args.out, clf, reg, le, scaler)
    else:
        interactive_mode(clf, reg, le, scaler)


if __name__ == "__main__":
    main()
