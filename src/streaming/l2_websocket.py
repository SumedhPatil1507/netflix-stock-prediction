"""Bridge normalized L2 websocket snapshots from a provider into Kafka.

Provider specific subscription/authentication is configured with environment
variables. Only full order-book messages (or explicitly normalized top of book)
are forwarded; ordinary trade/quote events are not mislabeled as L2 depth.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def normalize_l2_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize common Polygon/custom book snapshots to the Kafka schema."""
    ticker = event.get("ticker") or event.get("symbol") or event.get("sym") or event.get("S")
    bids, asks = event.get("bids"), event.get("asks")
    if bids and asks:
        bid = bids[0]
        ask = asks[0]
        bid_price = bid.get("price", bid.get("p")) if isinstance(bid, dict) else bid[0]
        bid_size = bid.get("size", bid.get("s")) if isinstance(bid, dict) else bid[1]
        ask_price = ask.get("price", ask.get("p")) if isinstance(ask, dict) else ask[0]
        ask_size = ask.get("size", ask.get("s")) if isinstance(ask, dict) else ask[1]
    elif all(key in event for key in ("bid_price", "bid_size", "ask_price", "ask_size")):
        bid_price, bid_size = event["bid_price"], event["bid_size"]
        ask_price, ask_size = event["ask_price"], event["ask_size"]
    else:
        return None
    timestamp = event.get("event_timestamp") or event.get("t") or event.get("timestamp")
    if isinstance(timestamp, (int, float)):
        # Polygon timestamps are commonly Unix milliseconds or nanoseconds.
        seconds = timestamp / (1e9 if timestamp > 1e14 else 1e3)
        timestamp = datetime.fromtimestamp(seconds, timezone.utc).isoformat()
    return {
        "ticker": str(ticker or "").upper(), "event_timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "bid_price": float(bid_price), "bid_size": float(bid_size),
        "ask_price": float(ask_price), "ask_size": float(ask_size),
        **({"trade_price": float(event["trade_price"]), "trade_size": float(event["trade_size"])}
           if "trade_price" in event and "trade_size" in event else {}),
    } if ticker else None


async def run_bridge() -> None:
    """Reconnect a configured websocket feed and publish normalized L2 to Kafka."""
    try:
        import websockets
        from aiokafka import AIOKafkaProducer
    except ImportError as exc:
        raise RuntimeError("Install requirements-streaming.txt to run the L2 bridge") from exc
    url = os.getenv("L2_WEBSOCKET_URL")
    if not url:
        raise RuntimeError("Set L2_WEBSOCKET_URL to the provider's websocket endpoint")
    producer = AIOKafkaProducer(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    await producer.start()
    topic = os.getenv("L2_KAFKA_TOPIC", "alpaca.l2")
    headers = {}
    if os.getenv("POLYGON_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['POLYGON_API_KEY']}"
    elif os.getenv("ALPACA_API_KEY"):
        headers.update({"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
                        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", "")})
    subscription = os.getenv("L2_WEBSOCKET_SUBSCRIBE")
    try:
        while True:
            try:
                async with websockets.connect(url, additional_headers=headers or None, ping_interval=20) as socket:
                    if subscription:
                        await socket.send(subscription)
                    async for raw in socket:
                        batch = json.loads(raw)
                        for event in batch if isinstance(batch, list) else [batch]:
                            if not isinstance(event, dict):
                                continue
                            normalized = normalize_l2_event(event)
                            if normalized is None:
                                continue
                            await producer.send_and_wait(topic, json.dumps(normalized).encode("utf-8"))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("L2 websocket disconnected; reconnecting")
                await asyncio.sleep(2)
    finally:
        await producer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bridge())
