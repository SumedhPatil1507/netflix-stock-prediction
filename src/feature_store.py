"""Feast definitions for point-in-time stock market features."""
from __future__ import annotations

try:
    from feast import Entity, FeatureView, Field, FileSource
    from feast.types import Float64, String
except ImportError as exc:  # Keep the base application usable without optional Feast.
    raise RuntimeError("Install feast to use the feature store definitions") from exc

stock = Entity(name="ticker", join_keys=["ticker"], value_type=String)
stock_features_source = FileSource(
    name="stock_features_source",
    path="data/feast/stock_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
stock_features = FeatureView(
    name="stock_metrics",
    entities=[stock],
    ttl=None,
    schema=[
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="ofi", dtype=Float64),
        Field(name="vwap", dtype=Float64),
        Field(name="vwap_micro_slippage_bps", dtype=Float64),
        Field(name="spread_bps", dtype=Float64),
    ],
    source=stock_features_source,
)
