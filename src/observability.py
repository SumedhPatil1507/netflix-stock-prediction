"""Shared OpenTelemetry spans and Prometheus metrics for runtime services."""
from __future__ import annotations

from contextlib import nullcontext
from threading import Lock
from typing import Any

try:
    from opentelemetry import trace
    tracer = trace.get_tracer("netflix-recommendations")
except ImportError:  # pragma: no cover - minimal local installations
    tracer = None

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

    registry = CollectorRegistry()
    candidate_retrieval_seconds = Histogram(
        "recommendation_candidate_retrieval_seconds", "Vector candidate retrieval latency", registry=registry)
    ranking_seconds = Histogram(
        "recommendation_ranking_seconds", "Contextual ranking latency", registry=registry)
    stream_bitrate_kbps = Gauge(
        "stream_bitrate_kbps", "Latest client-reported adaptive stream bitrate", ["rendition"], registry=registry)
    profile_cache_requests = Counter(
        "profile_cache_requests_total", "Redis profile lookup outcomes", ["result"], registry=registry)
    profile_cache_hit_ratio = Gauge(
        "profile_cache_hit_ratio", "Fraction of non-empty Redis profile history lookups", registry=registry)
    kafka_queue_depth = Gauge(
        "kafka_queue_depth", "Observed consumer lag in Kafka records", registry=registry)
    profile_cache_hit_ratio.set(0.0)
    kafka_queue_depth.set(0)
    _prometheus_available = True
except ImportError:  # pragma: no cover - API startup reports missing dependency
    registry = None
    candidate_retrieval_seconds = ranking_seconds = None
    stream_bitrate_kbps = profile_cache_requests = profile_cache_hit_ratio = kafka_queue_depth = None
    generate_latest = None
    _prometheus_available = False

_cache_counts_lock = Lock()
_cache_hits = 0
_cache_misses = 0
_tracing_configured = False


def span(name: str):
    """Return a named span context, even when OpenTelemetry is not installed."""
    return tracer.start_as_current_span(name) if tracer is not None else nullcontext()


def configure_tracing(service_name: str) -> bool:
    """Configure an OTLP trace exporter when an endpoint and SDK are available."""
    global _tracing_configured
    import os
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint or tracer is None or _tracing_configured:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _tracing_configured = True
        return True
    except ImportError:
        return False


def observe(metric: Any, value: float) -> None:
    if metric is not None:
        metric.observe(max(0.0, float(value)))


def set_stream_bitrate(rendition: str, bitrate_kbps: float) -> None:
    if stream_bitrate_kbps is not None:
        stream_bitrate_kbps.labels(rendition=rendition).set(max(0.0, float(bitrate_kbps)))


def record_cache_result(hit: bool) -> None:
    global _cache_hits, _cache_misses
    if profile_cache_requests is not None:
        with _cache_counts_lock:
            profile_cache_requests.labels(result="hit" if hit else "miss").inc()
            if hit:
                _cache_hits += 1
            else:
                _cache_misses += 1
            total = _cache_hits + _cache_misses
            if profile_cache_hit_ratio is not None:
                profile_cache_hit_ratio.set(_cache_hits / total if total else 0.0)


def set_kafka_queue_depth(depth: int) -> None:
    if kafka_queue_depth is not None:
        kafka_queue_depth.set(max(0, int(depth)))


def render_metrics() -> bytes:
    if not _prometheus_available:
        raise RuntimeError("Install prometheus-client to expose /metrics")
    return generate_latest(registry)
