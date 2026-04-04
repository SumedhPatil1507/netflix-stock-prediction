"""
FastAPI prediction service.
Run with: uvicorn api.main:app --reload
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.modeling import FEATURES

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Netflix Stock Prediction API",
    description="Predicts next-day return (%) and price for NFLX using a stacking ensemble.",
    version="1.0.0",
)

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    _model = None
    logger.error(f"Model load failed: {e}")


class OHLCVRow(BaseModel):
    open:   float = Field(..., example=645.0)
    high:   float = Field(..., example=655.0)
    low:    float = Field(..., example=640.0)
    close:  float = Field(..., example=650.0)
    volume: float = Field(..., example=5000000)


class PredictRequest(BaseModel):
    rows: List[OHLCVRow] = Field(..., min_items=10,
        description="Last N trading days of OHLCV data, most recent last")


class PredictResponse(BaseModel):
    predicted_return_pct: float
    predicted_next_close: float
    last_close:           float
    signal:               str   # "BUY" | "HOLD"


def _compute_features(rows: List[OHLCVRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.dict() for r in rows])
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    df["Return"]    = df["Close"].pct_change() * 100
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RangePct"]  = (df["High"] - df["Low"]) / df["Close"].shift(1) * 100

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"Lag{lag}"] = df["Close"].shift(lag)
    for lag in [1, 2, 3, 5]:
        df[f"RetLag{lag}"] = df["Return"].shift(lag)

    for w in [5, 7, 10, 21, 50, 200]:
        df[f"MA{w}"] = df["Close"].rolling(w, min_periods=1).mean()

    df["EMA9"]      = df["Close"].ewm(span=9,  adjust=False).mean()
    df["EMA21"]     = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_Cross"] = df["EMA9"] - df["EMA21"]

    df["RollingMean_5"]  = df["Close"].rolling(5,  min_periods=1).mean()
    df["RollingMean_10"] = df["Close"].rolling(10, min_periods=1).mean()
    df["RollingStd_5"]   = df["Close"].rolling(5,  min_periods=1).std().fillna(0)
    df["RollingStd_10"]  = df["Close"].rolling(10, min_periods=1).std().fillna(0)
    df["Volatility"]     = df["Return"].rolling(20, min_periods=1).std().fillna(0)
    df["Volatility_5"]   = df["Return"].rolling(5,  min_periods=1).std().fillna(0)

    for col in ["MA5","MA10","MA21","MA50","MA200"]:
        df[f"Price_vs_{col}"] = df["Close"] / df[col].replace(0, np.nan) - 1

    df["Volume_MA10"]    = df["Volume"].rolling(10, min_periods=1).mean()
    df["Volume_MA20"]    = df["Volume"].rolling(20, min_periods=1).mean()
    df["Volume_Ratio"]   = df["Volume"] / df["Volume_MA10"].replace(0, np.nan)
    df["Volume_Ratio20"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)

    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["OBV"]       = obv
    df["OBV_MA10"]  = df["OBV"].rolling(10, min_periods=1).mean()
    df["OBV_Ratio"] = df["OBV"] / df["OBV_MA10"].replace(0, np.nan)

    for period in [7, 14]:
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"RSI{period}"] = (100 - (100 / (1 + rs))).fillna(50)
    df["RSI"] = df["RSI14"]

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["MACD_Norm"]   = df["MACD"] / df["Close"].replace(0, np.nan)

    bb_mid         = df["Close"].rolling(20, min_periods=1).mean()
    bb_std         = df["Close"].rolling(20, min_periods=1).std().fillna(0)
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    denom          = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    df["BB_Width"] = denom / bb_mid
    df["BB_Pct"]   = (df["Close"] - df["BB_Lower"]) / denom

    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift(1)).abs()
    lc  = (df["Low"]  - df["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"]      = tr.rolling(14, min_periods=1).mean()
    df["ATR_Norm"] = df["ATR"] / df["Close"].replace(0, np.nan)

    low14  = df["Low"].rolling(14,  min_periods=1).min()
    high14 = df["High"].rolling(14, min_periods=1).max()
    rng14  = (high14 - low14).replace(0, np.nan)
    df["Stoch_K"]    = 100 * (df["Close"] - low14) / rng14
    df["Stoch_D"]    = df["Stoch_K"].rolling(3, min_periods=1).mean()
    df["Williams_R"] = -100 * (high14 - df["Close"]) / rng14

    tp     = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_ma  = tp.rolling(20, min_periods=1).mean()
    tp_std = tp.rolling(20, min_periods=1).std().replace(0, np.nan)
    df["CCI"] = (tp - tp_ma) / (0.015 * tp_std)

    df["Range"]      = df["High"] - df["Low"]
    df["Range_Norm"] = df["Range"] / df["Close"].replace(0, np.nan)
    df["Momentum5"]  = df["Close"] / df["Close"].shift(5).replace(0,  np.nan) - 1
    df["Momentum10"] = df["Close"] / df["Close"].shift(10).replace(0, np.nan) - 1
    df["Momentum20"] = df["Close"] / df["Close"].shift(20).replace(0, np.nan) - 1

    df["DayOfWeek"] = 2
    df["Month"]     = pd.Timestamp.now().month
    df["Quarter"]   = (pd.Timestamp.now().month - 1) // 3 + 1

    return df.ffill().fillna(0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python main.py first.")

    try:
        df = _compute_features(request.rows)
        row         = df[FEATURES].iloc[[-1]]
        pred_return = float(_model.predict(row)[0])
        last_close  = request.rows[-1].close
        pred_price  = last_close * (1 + pred_return / 100)
        signal      = "BUY" if pred_return > 0 else "HOLD"

        return PredictResponse(
            predicted_return_pct  = round(pred_return, 4),
            predicted_next_close  = round(pred_price, 2),
            last_close            = last_close,
            signal                = signal,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def features():
    return {"features": FEATURES, "count": len(FEATURES)}
