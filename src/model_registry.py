"""
Model versioning and registry.
Saves models with timestamps, maintains a registry JSON,
supports loading by version or latest.
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime

import joblib

logger = logging.getLogger(__name__)

MODELS_DIR    = "models"
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
LATEST_PATH   = os.path.join(MODELS_DIR, "model.pkl")  # always points to latest


def save_versioned_model(model, metrics: dict, ticker: str = "NFLX") -> str:
    """
    Save model with a versioned filename and update the registry.

    Returns the versioned model path.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"model_{ticker}_{ts}.pkl"
    path    = os.path.join(MODELS_DIR, version)

    joblib.dump(model, path)
    joblib.dump(model, LATEST_PATH)  # overwrite latest

    # Update registry
    registry = _load_registry()
    registry["models"].append({
        "version":    version,
        "path":       path,
        "ticker":     ticker,
        "trained_at": ts,
        "metrics":    metrics,
        "n_features": len(getattr(model, "feature_names_", [])),
    })
    registry["latest"] = version
    _save_registry(registry)

    logger.info(f"Model saved: {path}")
    logger.info(f"Registry updated: {len(registry['models'])} versions")
    return path


def load_latest_model():
    """Load the latest model from models/model.pkl."""
    return joblib.load(LATEST_PATH)


def load_model_by_version(version: str):
    """Load a specific model version from the registry."""
    path = os.path.join(MODELS_DIR, version)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model version not found: {version}")
    return joblib.load(path)


def get_registry() -> dict:
    """Return the full model registry."""
    return _load_registry()


def get_latest_version() -> str | None:
    """Return the latest model version string."""
    return _load_registry().get("latest")


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": [], "latest": None}


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
