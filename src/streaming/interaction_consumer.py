"""Manually-committed interaction consumer with a malformed-event DLQ."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from src.streaming.kafka_producer import DLQ_TOPIC, INTERACTION_TOPIC, InteractionEvent
from src.observability import set_kafka_queue_depth, span

logger = logging.getLogger(__name__)


class InteractionConsumer:
    """Validate events, forward valid records to a sink, and dead-letter invalid ones."""

    def __init__(self, *, consumer: Any = None, producer: Any = None, sink: Any = None):
        self.consumer = consumer
        self.producer = producer
        self.sink = sink
        self.redis = None
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

    async def start(self) -> None:
        from src.observability import configure_tracing
        configure_tracing("netflix-interaction-consumer")
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise RuntimeError("Install requirements-streaming.txt to run the interaction consumer") from exc
        if self.consumer is None:
            self.consumer = AIOKafkaConsumer(
                os.getenv("INTERACTIONS_KAFKA_TOPIC", INTERACTION_TOPIC),
                bootstrap_servers=self.bootstrap_servers,
                group_id=os.getenv("INTERACTIONS_CONSUMER_GROUP", "netflix-interactions-v1"),
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            )
        if self.producer is None:
            self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                             enable_idempotence=True)
        await self.consumer.start()
        await self.producer.start()
        if self.sink is None:
            try:
                from redis.asyncio import Redis
            except ImportError as exc:
                raise RuntimeError("Pass a sink or install redis for interaction persistence") from exc
            self.redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)

    async def handle(self, record: Any) -> bool:
        """Process one Kafka record, committing only after sink/DLQ delivery succeeds."""
        raw = record.value
        try:
            decoded = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            event = InteractionEvent.model_validate(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            dead_letter = {
                "source_topic": record.topic,
                "source_partition": record.partition,
                "source_offset": record.offset,
                "error": str(exc)[:1000],
                "raw_payload": raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw),
            }
            await self.producer.send_and_wait(
                os.getenv("TELEMETRY_DLQ_TOPIC", DLQ_TOPIC),
                json.dumps(dead_letter, separators=(",", ":")).encode("utf-8"),
            )
        else:
            payload = event.model_dump(mode="json")
            with span("kafka.interaction_consume"):
                if self.sink is not None:
                    result = self.sink(payload)
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    # Redis Streams provides durable downstream consumption.
                    encoded = json.dumps(payload, separators=(",", ":"))
                    await self.redis.xadd(os.getenv("INTERACTIONS_REDIS_STREAM", "netflix:user:interactions"),
                                          {"event": encoded},
                                          maxlen=int(os.getenv("INTERACTIONS_STREAM_MAXLEN", "100000")),
                                          approximate=True)
                    await self.redis.xadd(f"netflix:user:{event.user_id}:history", {"event": encoded},
                                          maxlen=int(os.getenv("USER_INTERACTION_HISTORY_MAXLEN", "10000")),
                                          approximate=True)
                    await self.redis.publish(f"netflix:recommendations:refresh:{event.user_id}", "refresh")
        await self.consumer.commit()
        await self._update_queue_depth()
        return True

    async def _update_queue_depth(self) -> None:
        """Export assigned-partition consumer lag when the client exposes watermarks."""
        if self.consumer is None or not all(hasattr(self.consumer, name)
                                             for name in ("assignment", "position", "highwater")):
            return
        try:
            depth = 0
            for partition in self.consumer.assignment():
                position = self.consumer.position(partition)
                if asyncio.iscoroutine(position):
                    position = await position
                highwater = self.consumer.highwater(partition)
                if highwater is not None:
                    depth += max(0, int(highwater) - int(position))
            set_kafka_queue_depth(depth)
        except Exception:
            logger.debug("Could not refresh Kafka consumer lag metric", exc_info=True)

    async def run(self) -> None:
        await self.start()
        try:
            async for record in self.consumer:
                try:
                    await self.handle(record)
                except Exception:
                    # Do not commit: Kafka will redeliver after restart/rebalance.
                    logger.exception("Interaction processing failed at %s:%s:%s",
                                     record.topic, record.partition, record.offset)
                    raise
        finally:
            await self.close()

    async def close(self) -> None:
        for client in (self.consumer, self.producer, self.redis):
            if client is not None:
                if client is self.redis:
                    await client.aclose()
                else:
                    await client.stop()


async def main() -> None:
    await InteractionConsumer().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
