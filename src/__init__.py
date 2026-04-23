"""
Netflix Stock Prediction — source package.
"""
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.feature_utils import compute_features_from_ohlcv, build_prediction_row
from src.modeling import train_model, save_model, load_model, FEATURES, get_active_features

__all__ = [
    "load_data", "preprocess_data", "create_features",
    "compute_features_from_ohlcv", "build_prediction_row",
    "train_model", "save_model", "load_model", "FEATURES", "get_active_features",
]
