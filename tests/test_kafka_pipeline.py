import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.streaming.interaction_consumer import InteractionConsumer
from src.streaming.kafka_producer import InteractionEvent, InteractionProducer


class FakeProducer:
    def __init__(self):
        self.messages = []

    async def send_and_wait(self, topic, value, **kwargs):
        self.messages.append((topic, value, kwargs))


class FakeConsumer:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


def test_producer_validates_and_publishes_versioned_interactions():
    async def run():
        producer = FakeProducer()
        service = InteractionProducer(producer=producer)
        payload = await service.publish({"event_type": "play", "user_id": "u-1", "content_id": "t-1"})
        assert payload["schema_version"] == 1
        assert payload["event_id"]
        assert producer.messages[0][0] == "netflix.user.interactions.v1"
        assert producer.messages[0][2]["key"] == b"u-1"
        with pytest.raises(ValidationError):
            await service.publish({"event_type": "unknown", "user_id": "u-1", "content_id": "t-1"})
    asyncio.run(run())


def test_consumer_persists_valid_event_before_committing():
    async def run():
        seen = []
        consumer = FakeConsumer()
        service = InteractionConsumer(consumer=consumer, sink=lambda event: seen.append(event))
        record = SimpleNamespace(topic="interactions", partition=0, offset=8,
            value=json.dumps({"event_type": "play", "user_id": "u-2", "content_id": "t-2"}).encode())
        assert await service.handle(record) is True
        assert len(seen) == 1 and seen[0]["content_id"] == "t-2"
        assert consumer.commits == 1
    asyncio.run(run())


def test_consumer_dead_letters_malformed_event_and_commits_after_delivery():
    async def run():
        consumer, producer = FakeConsumer(), FakeProducer()
        service = InteractionConsumer(consumer=consumer, producer=producer)
        record = SimpleNamespace(topic="interactions", partition=1, offset=12, value=b"not-json")
        assert await service.handle(record) is True
        topic, raw, _ = producer.messages[0]
        dead_letter = json.loads(raw)
        assert topic == "netflix.telemetry.dlq"
        assert dead_letter["raw_payload"] == "not-json"
        assert consumer.commits == 1
    asyncio.run(run())
