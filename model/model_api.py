# model/model_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
from pymongo import MongoClient
import numpy as np
import os

app = FastAPI(title="Stock Model API")

# --- CONFIG: edit if your files/uri differ ---
MODEL_PATH = "xgb_fwd_return_model.joblib"
SCALER_PATH = "xgb_fwd_return_scaler.joblib"
NSE_CSV = "NSE.csv"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB_NAME", "stockDB")
PRECOMP_COLL = "precomputed_daily"

# --- FEATURES used by your model (copied from model.py) ---
FEATURE_COLS = [
    "RSI",
    "SMA_20", "SMA_50",
    "EMA_12", "EMA_26",
    "MACD", "MACD_signal", "MACD_hist",
    "Volatility_20",
    "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    "Momentum_20",
    "ATR_14",
    "BB_Width",
    "ROC_10",
    "Stoch_K", "Stoch_D",
    "Williams_R",
    "OBV",
    "Volume_Z",
    "VWAP_Distance",
]

# --- Load model & scaler (must exist) ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

# --- Mongo client helper ---
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
precomp_coll = db[PRECOMP_COLL]

# --- Input schema ---
class PredictRequest(BaseModel):
    marketCap: Optional[str] = None    # e.g., "LARGE_CAP", "MID_CAP", "SMALL_CAP"
    riskTolerance: Optional[str] = None  # "Low"|"Medium"|"High"
    timeHorizonMonths: Optional[int] = None  # months
    topN: Optional[int] = 10

# --- Helper: fetch latest row for a symbol from precomputed_daily ---
def fetch_latest_features_for_symbol(symbol_full_name: str):
    # symbol_full_name = 'FULL COMPANY NAME' stored in your precomputed collection
    cur = precomp_coll.find({"symbol": symbol_full_name}).sort("date", -1).limit(1)
    row = next(cur, None)
    if not row:
        return None
    # Convert to pandas Series-like dict
    # Ensure feature keys exist; missing -> np.nan
    out = {}
    for c in FEATURE_COLS:
        out[c] = row.get(c, np.nan)
    return out

# --- Endpoint ---
@app.post("/predict-stocks")
def predict_stocks(req: PredictRequest):
    # 1) load NSE list and filter by cap
    try:
        nse_df = pd.read_csv(NSE_CSV)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load NSE.csv: {e}")

    # Normalize column names & required checks
    if "SYMBOL_NAME" not in nse_df.columns or "symbol" not in nse_df.columns or "cap_category" not in nse_df.columns:
        raise HTTPException(status_code=500, detail="NSE.csv must have 'symbol','SYMBOL_NAME','cap_category' columns")

    filtered = nse_df.copy()
    if req.marketCap:
        # allow partial matches like "LARGE" -> "LARGE_CAP"
        filtered = filtered[filtered["cap_category"].str.contains(req.marketCap, case=False, na=False)]

    symbols = filtered[["symbol", "SYMBOL_NAME"]].drop_duplicates().to_dict(orient="records")
    if len(symbols) == 0:
        raise HTTPException(status_code=400, detail="No symbols match the requested marketCap filter")

    # 2) For each symbol, fetch latest precomputed row & build feature vector
    rows = []
    for item in symbols:
        ticker = item["symbol"]
        fullname = item["SYMBOL_NAME"]
        feat = fetch_latest_features_for_symbol(fullname)
        if feat is None:
            continue
        feat["symbol"] = ticker
        rows.append(feat)

    if len(rows) == 0:
        raise HTTPException(status_code=404, detail="No precomputed data found in DB for filtered tickers")

    df_features = pd.DataFrame(rows)

    # 3) Ensure we only use columns present
    available = [c for c in FEATURE_COLS if c in df_features.columns]
    if len(available) == 0:
        raise HTTPException(status_code=500, detail="No required feature columns found in DB rows")

    # 4) Fill NaNs with 0 (or better strategy if you prefer)
    X = df_features[available].fillna(0).values

    # 5) Scale using scaler (if scaler expects full feature set, ensure same order)
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        # Attempt to pad missing features to scaler length if necessary
        raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

    # 6) Predict
    preds = model.predict(Xs)

    df_out = pd.DataFrame({
        "symbol": df_features["symbol"].values,
        "pred": preds
    })

    # Optionally apply a risk filter: for "Low", pick lower volatility stocks by Volatility_20 if available
    if req.riskTolerance and req.riskTolerance.lower() == "low" and "Volatility_20" in df_features.columns:
        df_out["vol"] = df_features["Volatility_20"].fillna(0).values
        # boost small volatility (penalize high vol)
        df_out = df_out.sort_values(["pred", "vol"], ascending=[False, True])

    df_out = df_out.sort_values("pred", ascending=False).head(req.topN).reset_index(drop=True)

    # Format result
    result = []
    for _, r in df_out.iterrows():
        result.append({
            "symbol": r["symbol"],
            "score": float(r["pred"])
        })

    return {"top": result, "count": len(result)}



# mock test


# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional

# app = FastAPI(title="Mock Stock Model API")

# class PredictRequest(BaseModel):
#     marketCap: Optional[str] = None
#     riskTolerance: Optional[str] = None
#     timeHorizonMonths: Optional[int] = None
#     topN: Optional[int] = 10

# # MOCK DATA - ALWAYS AVAILABLE
# MOCK_STOCKS = [
#     {"symbol": "TCS", "score": 0.83},
#     {"symbol": "INFY", "score": 0.75},
#     {"symbol": "RELIANCE", "score": 0.71},
#     {"symbol": "HDFCBANK", "score": 0.68},
#     {"symbol": "ICICIBANK", "score": 0.63},
#     {"symbol": "MARUTI", "score": 0.59},
#     {"symbol": "ITC", "score": 0.55},
# ]

# @app.post("/predict-stocks")
# def predict_stocks(req: PredictRequest):
#     # No DB, No CSV, No joblib — just mock predictions
#     return {
#         "filtersUsed": {
#             "marketCap": req.marketCap,
#             "riskTolerance": req.riskTolerance,
#             "timeHorizonMonths": req.timeHorizonMonths,
#         },
#         "top": MOCK_STOCKS[: req.topN],
#         "count": req.topN,
#     }