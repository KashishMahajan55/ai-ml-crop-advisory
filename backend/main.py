"""
backend/main.py
---------------
Synthetic dataset generated for educational AI/ML demonstration.

FastAPI backend — serves crop prediction and yield estimation.
Deploy this file to Render.com as a Web Service.

Endpoints:
  GET  /              — health check
  GET  /crops         — list of all supported crops
  POST /predict       — predict crop + yield from soil/climate inputs
  GET  /dataset/stats — summary statistics of the synthetic dataset
"""

import os
import sys
import pickle
import csv
import math

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── paths (backend/ is cwd on Render; go up one level for models/data) ───────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_CSV   = os.path.join(BASE_DIR, "data", "synthetic_crop_data.csv")

NOTE = "Synthetic dataset generated for educational AI/ML demonstration."

FEATURE_COLS = [
    "nitrogen_N", "phosphorus_P", "potassium_K",
    "temperature_C", "humidity_pct", "soil_pH", "rainfall_mm",
]

CROPS = [
    "Rice", "Wheat", "Maize", "Chickpea", "Lentil", "Cotton", "Sugarcane",
    "Soybean", "Groundnut", "Mungbean", "Blackgram", "Pigeonpea", "Jute",
    "Coffee", "Apple", "Mango", "Grapes", "Watermelon", "Papaya", "Coconut",
]

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI/ML Crop Advisory API",
    description=NOTE,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow requests from Vercel frontend and localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://*.vercel.app",
        "*",                        # tighten once you know your Vercel URL
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── load models once at startup ───────────────────────────────────────────────
def _load(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model artefact not found: {path}\n"
            "Run  python train_model.py  from the project root first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    clf    = _load("crop_recommender.pkl")
    reg    = _load("yield_estimator.pkl")
    le     = _load("label_encoder.pkl")
    scaler = _load("feature_scaler.pkl")
    models_loaded = True
except RuntimeError as e:
    print(f"[WARNING] {e}", file=sys.stderr)
    models_loaded = False

# ── request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    nitrogen_N:    float = Field(..., ge=0,   le=200,  example=90,  description="Nitrogen kg/ha")
    phosphorus_P:  float = Field(..., ge=0,   le=120,  example=45,  description="Phosphorus kg/ha")
    potassium_K:   float = Field(..., ge=0,   le=150,  example=40,  description="Potassium kg/ha")
    temperature_C: float = Field(..., ge=0,   le=50,   example=25,  description="Temperature °C")
    humidity_pct:  float = Field(..., ge=0,   le=100,  example=80,  description="Humidity %")
    soil_pH:       float = Field(..., ge=3.0, le=10.0, example=6.5, description="Soil pH")
    rainfall_mm:   float = Field(..., ge=0,   le=400,  example=180, description="Rainfall mm/month")

class CropResult(BaseModel):
    rank:           int
    crop:           str
    confidence_pct: float

class PredictResponse(BaseModel):
    top3:              list[CropResult]
    estimated_yield:   float
    yield_unit:        str
    dataset_note:      str

# ── serve frontend static files (no CORS needed — same origin) ───────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
def root():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>AI/ML Crop Advisory API</h2><p><a href='/docs'>API Docs</a></p>")

@app.get("/api/health", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "service":      "AI/ML Crop Advisory API",
        "dataset_note": NOTE,
        "docs":         "/docs",
    }

@app.get("/crops", tags=["Data"])
def get_crops():
    return {"crops": CROPS, "total": len(CROPS), "dataset_note": NOTE}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Check server logs.")

    features = [
        req.nitrogen_N, req.phosphorus_P, req.potassium_K,
        req.temperature_C, req.humidity_pct, req.soil_pH, req.rainfall_mm,
    ]
    X = np.array([features])
    X_s = scaler.transform(X)

    proba    = clf.predict_proba(X_s)[0]
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [
        CropResult(
            rank=i + 1,
            crop=le.classes_[idx],
            confidence_pct=round(float(proba[idx]) * 100, 1),
        )
        for i, idx in enumerate(top3_idx)
    ]
    yield_est = round(float(reg.predict(X_s)[0]), 2)

    return PredictResponse(
        top3=top3,
        estimated_yield=yield_est,
        yield_unit="tonnes/hectare",
        dataset_note=NOTE,
    )

@app.get("/dataset/stats", tags=["Data"])
def dataset_stats():
    if not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=404, detail="Dataset CSV not found on server.")
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    numeric = {c: [float(r[c]) for r in rows] for c in FEATURE_COLS + ["estimated_yield"]}
    stats = {}
    for col, vals in numeric.items():
        stats[col] = {
            "min":  round(min(vals), 2),
            "max":  round(max(vals), 2),
            "mean": round(sum(vals) / len(vals), 2),
        }
    crop_counts = {}
    for r in rows:
        crop_counts[r["crop_type"]] = crop_counts.get(r["crop_type"], 0) + 1
    return {
        "total_records": len(rows),
        "crop_counts":   crop_counts,
        "feature_stats": stats,
        "dataset_note":  NOTE,
    }
