"""Consume normalized Alpaca/depth websocket events from Kafka and publish features.

The websocket-to-Kafka bridge should normalize messages to ``ticker``,
``event_timestamp``, top-of-book prices/sizes, and optional trade fields. Alpaca's
stock websocket supplies trades and best quotes; full depth requires a feed that
provides L2 and should publish the same normalized schema.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.feature_utils import compute_l2_features

logger = logging.getLogger(__name__)


class L2FeatureConsumer:
    """Async Kafka consumer that computes causal features and writes Redis hashes."""

    def __init__(self, *, topic: str | None = None, bootstrap_servers: str | None = None,
                 redis_url: str | None = None, consumer: Any = None, redis_client: Any = None):
        self.topic = topic or os.getenv("L2_KAFKA_TOPIC", "alpaca.l2")
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.consumer = consumer
        self.redis = redis_client
        self._previous: dict[str, dict[str, float]] = {}
        self._totals: dict[str, tuple[float, float]] = {}

    async def start(self) -> None:
        if self.consumer is None:
            try:
                from aiokafka import AIOKafkaConsumer
            except ImportError as exc:
                raise RuntimeError("Install aiokafka to run the streaming consumer") from exc
            self.consumer = AIOKafkaConsumer(
                self.topic, bootstrap_servers=self.bootstrap_servers,
                group_id=os.getenv("KAFKA_CONSUMER_GROUP", "alpha-engine-features"),
                enable_auto_commit=False, value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
        if self.redis is None:
            try:
                from redis.asyncio import Redis
            except ImportError as exc:
                raise RuntimeError("Install redis to publish online features") from exc
            self.redis = Redis.from_url(self.redis_url, decode_responses=True)
        await self.consumer.start()

    async def process_message(self, message: dict[str, Any]) -> dict[str, float]:
        ticker = str(message["ticker"]).upper()
        bid_ask = {key: float(message[key]) for key in ("bid_price", "bid_size", "ask_price", "ask_size")}
        prior = self._previous.get(ticker)
        notional, volume = self._totals.get(ticker, (0.0, 0.0))
        features = compute_l2_features(message, prior, notional, volume)
        self._previous[ticker] = bid_ask
        self._totals[ticker] = (features["cumulative_notional"], features["cumulative_volume"])
        timestamp = message.get("event_timestamp") or datetime.now(timezone.utc).isoformat()
        payload = {key: str(value) for key, value in features.items() if not key.startswith("cumulative_")}
        payload["event_timestamp"] = str(timestamp)
        await self.redis.hset(f"alpha:features:{ticker}", mapping=payload)
        await self.redis.expire(f"alpha:features:{ticker}", int(os.getenv("FEATURE_TTL_SECONDS", "300")))
        return features

    async def run(self) -> None:
        await self.start()
        try:
            async for record in self.consumer:
                try:
                    message = record.value if isinstance(record.value, dict) else json.loads(record.value)
                    await self.process_message(message)
                    await self.consumer.commit()
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    logger.exception("Skipping malformed L2 event at offset %s", record.offset)
        finally:
            await self.close()

    async def close(self) -> None:
        if self.consumer is not None:
            await self.consumer.stop()
        if self.redis is not None:
            close = getattr(self.redis, "aclose", None) or getattr(self.redis, "close", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result


async def main() -> None:
    await L2FeatureConsumer().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
