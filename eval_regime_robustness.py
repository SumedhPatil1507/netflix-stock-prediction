#!/usr/bin/env python
"""
Backtest robustness report: performance across HMM-detected market regimes.
Run: python eval_regime_robustness.py
Outputs: outputs/eval_regime_robustness.json
"""
from __future__ import annotations
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    from src.data_loader import load_data
    from src.preprocessing import preprocess_data
    from src.feature_engineering import create_features
    from src.modeling import load_model, get_active_features
    from src.feature_utils import build_prediction_row  # noqa: F401 — imported for side-effects

    df_raw = load_data(source="csv")
    df = preprocess_data(df_raw)
    df = create_features(df)
    model = load_model()
    features = get_active_features(df)

    results = {}
    regime_col = next((c for c in ["Regime", "regime"] if c in df.columns), None)

    if regime_col is None:
        print("No regime column found — reporting overall.")
        regimes = {"all": df}
    else:
        regime_map = {0: "bear", 1: "sideways", 2: "bull"}
        regimes = {}
        for code, label in regime_map.items():
            sub = df[df[regime_col] == code]
            if len(sub) > 50:
                regimes[label] = sub

    for regime_name, sub_df in regimes.items():
        avail = [f for f in features if f in sub_df.columns]
        X = sub_df[avail].dropna()
        y = sub_df.loc[X.index, "Return"].shift(-1).dropna()
        X = X.loc[y.index]
        if len(X) < 10:
            continue

        preds = model.predict(X)
        actual = y.values
        errors = actual - preds
        direction_acc = float(np.mean(np.sign(preds) == np.sign(actual)) * 100)
        rmse = float(np.sqrt(np.mean(errors**2)))
        results[regime_name] = {
            "n": len(X),
            "dir_acc_pct": round(direction_acc, 2),
            "rmse": round(rmse, 4),
            "mean_pred": round(float(np.mean(preds)), 4),
            "mean_actual": round(float(np.mean(actual)), 4),
        }
        print(f"  {regime_name}: n={len(X)}, dir_acc={direction_acc:.1f}%, rmse={rmse:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_regime_robustness.json", "w") as f:
        json.dump({"regime_performance": results}, f, indent=2)
    print("\nSaved to outputs/eval_regime_robustness.json")


if __name__ == "__main__":
    main()
