"""Langfuse and local JSONL tracing for Market Narrator agent runs.

Local traces are always written to ``logs/agent_traces.jsonl``. When both
LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are configured, the same agent
execution is also emitted as a Langfuse span.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import uuid
from typing import Any, Iterator

logger = logging.getLogger(__name__)
TRACE_PATH = Path(__file__).resolve().parent / "logs" / "agent_traces.jsonl"


class AgentTrace:
    """Small handle used by agent nodes to attach their final output."""

    def __init__(self) -> None:
        self.output: Any = None
        self.metadata: dict[str, Any] = {}

    def set_output(self, output: Any, **metadata: Any) -> None:
        self.output = output
        self.metadata.update(metadata)


def _append_local(record: dict[str, Any]) -> None:
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        logger.exception("Could not write local agent trace to %s", TRACE_PATH)


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@contextmanager
def trace_agent_run(
    agent_name: str,
    ticker: str,
    inputs: Any,
    *,
    run_id: str | None = None,
) -> Iterator[AgentTrace]:
    """Trace one retrieval or synthesis node, without making tracing a hard dependency."""
    run_id = run_id or str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    trace = AgentTrace()
    status = "success"
    error = None
    langfuse = None
    observation_context = None
    observation = None

    if _langfuse_enabled():
        try:
            from langfuse import get_client

            langfuse = get_client()
            observation_context = langfuse.start_as_current_observation(
                as_type="span", name=f"market-narrator.{agent_name}"
            )
            observation = observation_context.__enter__()
            observation.update(
                input=inputs,
                metadata={"agent": agent_name, "ticker": ticker, "run_id": run_id},
            )
        except Exception:
            logger.exception("Langfuse span could not be started; keeping local trace")
            observation_context = None
            observation = None

    try:
        yield trace
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        if observation is not None:
            try:
                observation.update(level="ERROR", status_message=error)
            except Exception:
                logger.debug("Could not mark Langfuse observation as failed", exc_info=True)
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        record = {
            "event": "agent_run",
            "run_id": run_id,
            "agent": agent_name,
            "ticker": ticker,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "status": status,
            "inputs": inputs,
            "output": trace.output,
            "metadata": trace.metadata,
            "error": error,
        }
        _append_local(record)
        if observation is not None:
            try:
                observation.update(
                    output=trace.output,
                    metadata={
                        "agent": agent_name,
                        "ticker": ticker,
                        "run_id": run_id,
                        "status": status,
                        **trace.metadata,
                    },
                )
            except Exception:
                logger.exception("Could not update Langfuse observation")
        if observation_context is not None:
            try:
                observation_context.__exit__(None, None, None)
            except Exception:
                logger.exception("Could not close Langfuse observation")
        if langfuse is not None:
            try:
                langfuse.flush()
            except Exception:
                logger.debug("Could not flush Langfuse client", exc_info=True)
