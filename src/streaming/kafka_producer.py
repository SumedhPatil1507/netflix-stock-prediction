"""Async producer for versioned user interaction telemetry."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from src.observability import span

INTERACTION_TOPIC = "netflix.user.interactions.v1"
DLQ_TOPIC = "netflix.telemetry.dlq"
EVENT_TYPES = {"play", "pause", "seek", "abandon", "hover", "thumbs_up", "thumbs_down", "impression"}


class InteractionEvent(BaseModel):
    """Validated, privacy-minimal interaction event sent to Kafka."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: Literal["play", "pause", "seek", "abandon", "hover", "thumbs_up", "thumbs_down", "impression"]
    user_id: str = Field(min_length=1, max_length=128)
    content_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    position_seconds: float | None = Field(default=None, ge=0)
    session_id: str | None = Field(default=None, max_length=128)
    device_type: str | None = Field(default=None, max_length=32)
    session_duration_seconds: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    schema_version: Literal[1] = 1


class InteractionProducer:
    """Lazy-started aiokafka producer; injectable for callers and tests."""

    def __init__(self, producer: Any = None, bootstrap_servers: str | None = None):
        self.producer = producer
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    async def start(self) -> None:
        if self.producer is None:
            try:
                from aiokafka import AIOKafkaProducer
            except ImportError as exc:
                raise RuntimeError("Install requirements-streaming.txt to enable Kafka") from exc
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                enable_idempotence=True,
                value_serializer=lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8"),
            )
        await self.producer.start()

    async def publish(self, event: InteractionEvent | dict[str, Any]) -> dict[str, Any]:
        validated = event if isinstance(event, InteractionEvent) else InteractionEvent.model_validate(event)
        payload = validated.model_dump(mode="json")
        with span("kafka.interaction_publish"):
            await self.producer.send_and_wait(
                os.getenv("INTERACTIONS_KAFKA_TOPIC", INTERACTION_TOPIC),
                payload,
                key=validated.user_id.encode("utf-8"),
            )
        return payload

    async def close(self) -> None:
        if self.producer is not None:
            await self.producer.stop()

