"""
Model drift detection using Population Stability Index (PSI)
and Kolmogorov-Smirnov test.
Monitors whether live data distribution has shifted from training.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  : no significant change
    PSI 0.1-0.2: moderate change, monitor
    PSI > 0.2  : significant shift, retrain
    """
    eps = 1e-6
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + eps

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _ks_test(expected: np.ndarray, actual: np.ndarray) -> dict:
    """KS test — returns statistic and p-value."""
    from scipy import stats
    stat, pval = stats.ks_2samp(expected, actual)
    return {"ks_stat": float(stat), "ks_pval": float(pval)}


def detect_drift(train_df: pd.DataFrame,
                 live_df: pd.DataFrame,
                 features: list,
                 psi_threshold: float = 0.2) -> Dict:
    """
    Compare feature distributions between training and live data.

    Returns
    -------
    dict with:
      - per_feature: {feature: {psi, ks_stat, ks_pval, drifted}}
      - drifted_features: list of features with PSI > threshold
      - overall_drift: bool
    """
    results = {}
    drifted = []

    for feat in features:
        if feat not in train_df.columns or feat not in live_df.columns:
            continue
        tr  = train_df[feat].dropna().values
        lv  = live_df[feat].dropna().values
        if len(tr) < 10 or len(lv) < 10:
            continue

        psi_val = _psi(tr, lv)
        ks      = _ks_test(tr, lv)
        is_drift = psi_val > psi_threshold

        results[feat] = {
            "psi":     round(psi_val, 4),
            "ks_stat": round(ks["ks_stat"], 4),
            "ks_pval": round(ks["ks_pval"], 4),
            "drifted": is_drift,
        }
        if is_drift:
            drifted.append(feat)

    overall = len(drifted) > len(features) * 0.3  # >30% features drifted

    logger.info(f"Drift check: {len(drifted)}/{len(results)} features drifted")
    if overall:
        logger.warning("Significant overall drift detected — consider retraining")

    return {
        "per_feature":      results,
        "drifted_features": drifted,
        "overall_drift":    overall,
        "psi_threshold":    psi_threshold,
    }


def drift_summary_df(drift_result: dict) -> pd.DataFrame:
    """Convert drift result to a sorted DataFrame for display."""
    rows = [
        {"Feature": k, "PSI": v["psi"], "KS_Stat": v["ks_stat"],
         "KS_Pval": v["ks_pval"], "Drifted": v["drifted"]}
        for k, v in drift_result["per_feature"].items()
    ]
    df = pd.DataFrame(rows).sort_values("PSI", ascending=False)
    return df.reset_index(drop=True)
