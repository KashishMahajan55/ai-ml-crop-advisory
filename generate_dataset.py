"""
generate_dataset.py
-------------------
Synthetic dataset generated for educational AI/ML demonstration.

Generates a completely artificial agricultural dataset with no real farm data,
no personal information, and no connection to any real-world measurements.
The data is statistically plausible but entirely fictional.

Usage:
    python generate_dataset.py

Output:
    data/synthetic_crop_data.csv   — main dataset (3 000 records)
    data/README_data.txt           — dataset provenance note

Reproducibility:
    Fixed random seed (SEED = 42) ensures identical output on every run.
"""

import os
import random
import math
import csv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42           # Fixed seed → fully reproducible
N_RECORDS = 3_000   # Total synthetic records
OUTPUT_DIR = "data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "synthetic_crop_data.csv")
OUTPUT_NOTE = os.path.join(OUTPUT_DIR, "README_data.txt")

random.seed(SEED)

# ---------------------------------------------------------------------------
# Crop profiles — realistic agronomic ranges (all synthetic / educational)
# Each profile defines plausible ranges for N, P, K, Temp, Humidity, pH,
# Rainfall, and a base yield range (tonnes/hectare).
# ---------------------------------------------------------------------------

CROP_PROFILES = {
    "Rice": {
        "N": (60, 120), "P": (30, 60), "K": (30, 60),
        "Temp": (20, 35), "Humidity": (70, 95), "pH": (5.5, 7.0),
        "Rainfall": (150, 300), "Yield": (2.5, 6.0),
    },
    "Wheat": {
        "N": (80, 140), "P": (40, 70), "K": (30, 70),
        "Temp": (10, 25), "Humidity": (40, 70), "pH": (6.0, 7.5),
        "Rainfall": (50, 120), "Yield": (2.0, 5.5),
    },
    "Maize": {
        "N": (70, 130), "P": (35, 65), "K": (35, 65),
        "Temp": (18, 30), "Humidity": (50, 80), "pH": (5.8, 7.2),
        "Rainfall": (60, 150), "Yield": (3.0, 8.0),
    },
    "Chickpea": {
        "N": (20, 50), "P": (40, 80), "K": (30, 60),
        "Temp": (15, 30), "Humidity": (30, 60), "pH": (6.0, 8.0),
        "Rainfall": (30, 80), "Yield": (0.8, 2.5),
    },
    "Lentil": {
        "N": (15, 45), "P": (35, 75), "K": (25, 55),
        "Temp": (12, 28), "Humidity": (30, 65), "pH": (6.0, 8.0),
        "Rainfall": (25, 75), "Yield": (0.7, 2.2),
    },
    "Cotton": {
        "N": (80, 150), "P": (40, 80), "K": (40, 80),
        "Temp": (22, 38), "Humidity": (45, 75), "pH": (5.8, 7.8),
        "Rainfall": (60, 130), "Yield": (1.0, 3.5),
    },
    "Sugarcane": {
        "N": (100, 180), "P": (40, 80), "K": (60, 100),
        "Temp": (24, 38), "Humidity": (65, 90), "pH": (6.0, 7.5),
        "Rainfall": (120, 250), "Yield": (40.0, 100.0),
    },
    "Soybean": {
        "N": (20, 60), "P": (35, 70), "K": (35, 70),
        "Temp": (18, 32), "Humidity": (50, 80), "pH": (6.0, 7.5),
        "Rainfall": (60, 150), "Yield": (1.5, 4.0),
    },
    "Groundnut": {
        "N": (20, 55), "P": (40, 80), "K": (30, 60),
        "Temp": (22, 35), "Humidity": (45, 75), "pH": (5.5, 7.5),
        "Rainfall": (50, 120), "Yield": (1.2, 3.5),
    },
    "Mungbean": {
        "N": (15, 45), "P": (30, 65), "K": (25, 55),
        "Temp": (20, 35), "Humidity": (55, 85), "pH": (6.0, 7.5),
        "Rainfall": (40, 100), "Yield": (0.6, 1.8),
    },
    "Blackgram": {
        "N": (15, 45), "P": (30, 65), "K": (25, 55),
        "Temp": (22, 35), "Humidity": (60, 85), "pH": (5.5, 7.5),
        "Rainfall": (45, 110), "Yield": (0.6, 1.8),
    },
    "Pigeonpea": {
        "N": (20, 50), "P": (35, 70), "K": (30, 60),
        "Temp": (20, 35), "Humidity": (50, 80), "pH": (5.5, 7.5),
        "Rainfall": (60, 140), "Yield": (0.8, 2.5),
    },
    "Jute": {
        "N": (60, 110), "P": (30, 60), "K": (30, 60),
        "Temp": (24, 37), "Humidity": (70, 95), "pH": (6.0, 7.8),
        "Rainfall": (150, 300), "Yield": (1.5, 4.0),
    },
    "Coffee": {
        "N": (80, 130), "P": (35, 70), "K": (35, 70),
        "Temp": (16, 28), "Humidity": (65, 90), "pH": (5.5, 6.5),
        "Rainfall": (150, 300), "Yield": (0.5, 2.5),
    },
    "Apple": {
        "N": (60, 110), "P": (30, 65), "K": (30, 65),
        "Temp": (5, 20), "Humidity": (50, 80), "pH": (5.5, 7.0),
        "Rainfall": (60, 150), "Yield": (10.0, 40.0),
    },
    "Mango": {
        "N": (50, 110), "P": (25, 60), "K": (30, 70),
        "Temp": (24, 40), "Humidity": (55, 85), "pH": (5.5, 7.5),
        "Rainfall": (60, 150), "Yield": (5.0, 20.0),
    },
    "Grapes": {
        "N": (50, 110), "P": (30, 65), "K": (35, 75),
        "Temp": (15, 35), "Humidity": (50, 80), "pH": (5.5, 7.0),
        "Rainfall": (40, 100), "Yield": (8.0, 30.0),
    },
    "Watermelon": {
        "N": (50, 100), "P": (30, 65), "K": (30, 70),
        "Temp": (22, 38), "Humidity": (55, 85), "pH": (6.0, 7.0),
        "Rainfall": (40, 100), "Yield": (15.0, 50.0),
    },
    "Papaya": {
        "N": (60, 120), "P": (30, 70), "K": (40, 80),
        "Temp": (22, 38), "Humidity": (60, 90), "pH": (6.0, 7.5),
        "Rainfall": (80, 200), "Yield": (20.0, 60.0),
    },
    "Coconut": {
        "N": (70, 130), "P": (30, 65), "K": (60, 120),
        "Temp": (24, 38), "Humidity": (70, 95), "pH": (5.5, 8.0),
        "Rainfall": (100, 250), "Yield": (50.0, 150.0),
    },
}

CROPS = list(CROP_PROFILES.keys())

# ---------------------------------------------------------------------------
# Helper: uniform float with 2 decimal places
# ---------------------------------------------------------------------------

def randf(lo, hi):
    return round(random.uniform(lo, hi), 2)


def add_noise(value, pct=0.05):
    """Add ±pct Gaussian-like noise; clamp to 0."""
    delta = value * pct * (2 * random.random() - 1)
    return round(max(0.0, value + delta), 2)


# ---------------------------------------------------------------------------
# Synthetic yield model
# A simple rule: yield scales linearly with how well nutrient levels sit in
# the ideal mid-range.  Pure fiction for demonstration purposes.
# ---------------------------------------------------------------------------

def synthetic_yield(crop, n, p, k, temp, humidity, rainfall):
    profile = CROP_PROFILES[crop]
    ylo, yhi = profile["Yield"]

    def mid_score(val, lo, hi):
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2
        if spread == 0:
            return 1.0
        return max(0.0, 1.0 - abs(val - mid) / spread)

    score = (
        mid_score(n,        *profile["N"]) * 0.20 +
        mid_score(p,        *profile["P"]) * 0.15 +
        mid_score(k,        *profile["K"]) * 0.15 +
        mid_score(temp,     *profile["Temp"]) * 0.20 +
        mid_score(humidity, *profile["Humidity"]) * 0.15 +
        mid_score(rainfall, *profile["Rainfall"]) * 0.15
    )
    base_yield = ylo + score * (yhi - ylo)
    return add_noise(base_yield, pct=0.08)


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "record_id", "crop_type",
    "nitrogen_N", "phosphorus_P", "potassium_K",
    "temperature_C", "humidity_pct", "soil_pH", "rainfall_mm",
    "estimated_yield",
    "dataset_note",
]

NOTE = "Synthetic dataset generated for educational AI/ML demonstration."


def generate_records(n):
    records = []
    for i in range(1, n + 1):
        crop = random.choice(CROPS)
        p = CROP_PROFILES[crop]

        n_val  = randf(*p["N"])
        p_val  = randf(*p["P"])
        k_val  = randf(*p["K"])
        t_val  = randf(*p["Temp"])
        h_val  = randf(*p["Humidity"])
        ph_val = randf(*p["pH"])
        r_val  = randf(*p["Rainfall"])
        y_val  = synthetic_yield(crop, n_val, p_val, k_val, t_val, h_val, r_val)

        records.append({
            "record_id":        i,
            "crop_type":        crop,
            "nitrogen_N":       n_val,
            "phosphorus_P":     p_val,
            "potassium_K":      k_val,
            "temperature_C":    t_val,
            "humidity_pct":     h_val,
            "soil_pH":          ph_val,
            "rainfall_mm":      r_val,
            "estimated_yield":  y_val,
            "dataset_note":     NOTE,
        })
    return records


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_csv(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] CSV written: {path}  ({len(records)} records)")


def write_provenance(path):
    text = (
        "DATASET PROVENANCE\n"
        "==================\n\n"
        "File:    synthetic_crop_data.csv\n"
        "Records: 3 000\n"
        "Seed:    42 (reproducible)\n\n"
        "This dataset is ENTIRELY SYNTHETIC.\n"
        "It was programmatically generated by generate_dataset.py\n"
        "for educational AI/ML demonstration purposes ONLY.\n\n"
        "It does NOT represent:\n"
        "  - Real farm measurements\n"
        "  - Real agricultural records\n"
        "  - Any specific geographic region\n"
        "  - Any real farmer or organisation\n\n"
        "It does NOT contain:\n"
        "  - Personal names\n"
        "  - Phone numbers or email addresses\n"
        "  - Physical or GPS addresses\n"
        "  - Identity numbers (Aadhaar, PAN, etc.)\n"
        "  - Any private or sensitive information\n\n"
        "Source:  generate_dataset.py (this repository)\n"
        "License: For educational/demonstration use only.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] Provenance note written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Synthetic Agricultural Dataset Generator")
    print(NOTE)
    print("=" * 60)
    records = generate_records(N_RECORDS)
    write_csv(records, OUTPUT_CSV)
    write_provenance(OUTPUT_NOTE)
    print("Done. Run train_model.py to train the ML models.")
