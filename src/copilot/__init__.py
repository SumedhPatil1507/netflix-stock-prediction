"""
Research Copilot package — Alpha Engine Pro.

Provides an agentic RAG pipeline:
    CopilotRetriever → CopilotToolAgent → CopilotWriter → HITLRouter

Orchestrated by :class:`CopilotGraph`, which uses LangGraph StateGraph
when available and falls back to a sequential pipeline otherwise.

Typical usage
-------------
    from src.copilot import CopilotGraph

    g = CopilotGraph(ticker="NFLX")
    result = g.run("NFLX")
    print(result["note"])
"""
from __future__ import annotations

import logging

# Graceful import guards — each sub-module catches its own optional deps.
try:
    from src.copilot.retriever import CopilotRetriever
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("CopilotRetriever unavailable: %s", _e)
    CopilotRetriever = None  # type: ignore[assignment,misc]

try:
    from src.copilot.tool_agent import CopilotToolAgent
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("CopilotToolAgent unavailable: %s", _e)
    CopilotToolAgent = None  # type: ignore[assignment,misc]

try:
    from src.copilot.writer import CopilotWriter
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("CopilotWriter unavailable: %s", _e)
    CopilotWriter = None  # type: ignore[assignment,misc]

try:
    from src.copilot.hitl_router import HITLRouter
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("HITLRouter unavailable: %s", _e)
    HITLRouter = None  # type: ignore[assignment,misc]

try:
    from src.copilot.graph import CopilotGraph, CopilotState
except Exception as _e:  # pragma: no cover
    logging.getLogger(__name__).warning("CopilotGraph unavailable: %s", _e)
    CopilotGraph = None  # type: ignore[assignment,misc]
    CopilotState = None  # type: ignore[assignment,misc]

__all__ = [
    "CopilotRetriever",
    "CopilotToolAgent",
    "CopilotWriter",
    "HITLRouter",
    "CopilotGraph",
    "CopilotState",
]
