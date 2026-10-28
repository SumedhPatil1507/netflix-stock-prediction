"""Point-in-time feature joins must never expose future or late-arriving values."""
import pandas as pd

from src.data_loader import point_in_time_join


def test_point_in_time_join_uses_latest_available_observation():
    entities = pd.DataFrame({
        "ticker": ["NFLX", "NFLX"],
        "event_timestamp": ["2024-01-01T10:02:00Z", "2024-01-01T10:05:00Z"],
    })
    features = pd.DataFrame({
        "ticker": ["NFLX", "NFLX", "NFLX"],
        "event_timestamp": ["2024-01-01T10:00:00Z", "2024-01-01T10:03:00Z", "2024-01-01T10:04:00Z"],
        "created_timestamp": ["2024-01-01T10:00:01Z", "2024-01-01T10:06:00Z", "2024-01-01T10:04:01Z"],
        "ofi": [1.0, 999.0, 4.0],
    })

    result = point_in_time_join(entities, features)

    assert result["ofi"].tolist() == [1.0, 4.0]
    assert result["feature_event_timestamp"].iloc[0] == pd.Timestamp("2024-01-01T10:00:00Z")


def test_point_in_time_join_has_no_future_or_late_data_leakage():
    entities = pd.DataFrame({
        "ticker": ["NFLX"],
        "event_timestamp": ["2024-01-01T10:02:00Z"],
    })
    features = pd.DataFrame({
        "ticker": ["NFLX", "NFLX"],
        "event_timestamp": ["2024-01-01T10:01:00Z", "2024-01-01T10:03:00Z"],
        "created_timestamp": ["2024-01-01T10:04:00Z", "2024-01-01T10:03:01Z"],
        "ofi": [123.0, 456.0],
    })

    result = point_in_time_join(entities, features)

    assert pd.isna(result.loc[0, "ofi"])
    assert pd.isna(result.loc[0, "feature_created_timestamp"])
