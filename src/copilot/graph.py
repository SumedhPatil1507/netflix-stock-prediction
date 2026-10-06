"""
CopilotGraph — orchestrates the Research Copilot pipeline.

Runs four stages in sequence:
    retrieve → tool → write → route

Uses LangGraph StateGraph when available; falls back to a plain sequential
call chain otherwise.  Always returns a dict and never raises.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional LangGraph ────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END  # type: ignore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.info("LangGraph not installed — CopilotGraph will run in sequential mode.")

# ── State TypedDict ───────────────────────────────────────────────────────────
from typing import TypedDict, Optional


class CopilotState(TypedDict):
    """State passed between nodes in the LangGraph (or sequential) pipeline."""

    ticker: str
    query: Optional[str]
    tool_state: dict[str, Any]
    retrieval: dict[str, Any]
    note: str
    hitl_result: dict[str, Any]
    error: Optional[str]


# ── Main class ────────────────────────────────────────────────────────────────

class CopilotGraph:
    """
    End-to-end Research Copilot pipeline.

    Parameters
    ----------
    ticker             : Default ticker symbol.
    hitl_threshold_usd : USD position-value threshold for HITL gating.
    """

    def __init__(
        self,
        ticker: str = "NFLX",
        hitl_threshold_usd: float = 10_000.0,
    ) -> None:
        self.ticker = ticker
        self.hitl_threshold_usd = hitl_threshold_usd

        # Lazy-import sub-components to avoid circular imports
        from src.copilot.retriever import CopilotRetriever
        from src.copilot.tool_agent import CopilotToolAgent
        from src.copilot.writer import CopilotWriter
        from src.copilot.hitl_router import HITLRouter, make_signal_id

        self._retriever = CopilotRetriever(ticker=ticker)
        self._tool_agent = CopilotToolAgent()
        self._writer = CopilotWriter()
        self._hitl = HITLRouter(threshold_usd=hitl_threshold_usd)
        self._make_signal_id = make_signal_id

        # Build LangGraph if available
        self._graph = None
        if LANGGRAPH_AVAILABLE:
            try:
                self._graph = self._build_graph()
                logger.info("CopilotGraph initialised with LangGraph StateGraph.")
            except Exception as exc:
                logger.warning("LangGraph StateGraph build failed (%s); using sequential mode.", exc)
        else:
            logger.info("CopilotGraph initialised in sequential mode.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        ticker: str | None = None,
        query: str | None = None,
        position_value: float = 0.0,
    ) -> dict[str, Any]:
        """
        Execute the full copilot pipeline.

        Parameters
        ----------
        ticker         : Override the default ticker.
        query          : Custom retrieval query (auto-generated if None).
        position_value : USD value of the proposed position (for HITL gating).

        Returns
        -------
        dict with keys:
            ``tool_state``, ``retrieval`` (dict with ``documents``),
            ``note`` (research note string), ``hitl_result``,
            ``sources_used`` (int), ``error`` (str | None).
        Never raises.
        """
        _ticker = ticker or self.ticker
        try:
            if LANGGRAPH_AVAILABLE and self._graph is not None:
                return self._run_graph(_ticker, query, position_value)
            return self._run_sequential(_ticker, query, position_value)
        except Exception as exc:
            logger.error("CopilotGraph.run failed: %s", exc)
            return {
                "tool_state": {},
                "retrieval": {"documents": []},
                "note": f"[Research Copilot error: {exc}]",
                "hitl_result": {},
                "sources_used": 0,
                "error": str(exc),
            }

    # ── LangGraph path ─────────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph StateGraph."""
        workflow: StateGraph = StateGraph(CopilotState)  # type: ignore[type-arg]

        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("tool", self._node_tool)
        workflow.add_node("write", self._node_write)
        workflow.add_node("route", self._node_route)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "tool")
        workflow.add_edge("tool", "write")
        workflow.add_edge("write", "route")
        workflow.add_edge("route", END)

        return workflow.compile()

    def _run_graph(
        self,
        ticker: str,
        query: str | None,
        position_value: float,
    ) -> dict[str, Any]:
        """Invoke the compiled LangGraph StateGraph."""
        initial: CopilotState = {
            "ticker": ticker,
            "query": query,
            "tool_state": {},
            "retrieval": {},
            "note": "",
            "hitl_result": {},
            "error": None,
        }
        # Carry position_value via query string for the route node
        initial["_position_value"] = position_value  # type: ignore[typeddict-unknown-key]

        result = self._graph.invoke(initial)
        return self._state_to_output(result)

    # ── Node implementations (shared with sequential path) ─────────────────────

    def _node_retrieve(self, state: CopilotState) -> CopilotState:
        ticker = state["ticker"]
        query = state.get("query") or f"{ticker} stock analysis earnings signal"
        try:
            # Auto-ingest if collection is empty
            stats = self._retriever.vector_store.get_collection_stats()
            if stats.get("document_count", 0) == 0:
                self._retriever.ingest_ticker(ticker)

            retrieval = self._retriever.retrieve(query=query, ticker=ticker)
        except Exception as exc:
            logger.error("_node_retrieve failed: %s", exc)
            retrieval = {"query": query, "ticker": ticker, "documents": []}
            state["error"] = str(exc)

        state["retrieval"] = retrieval
        state["query"] = query
        return state

    def _node_tool(self, state: CopilotState) -> CopilotState:
        ticker = state["ticker"]
        try:
            tool_state = self._tool_agent.get_model_state(ticker=ticker)
            # Attach SHAP drivers directly into tool_state for writer convenience
            if tool_state:
                tool_state["shap_drivers"] = self._tool_agent.get_shap_drivers(n=5)
                risk = self._tool_agent.get_risk_state(
                    ticker=ticker,
                    pred_return=tool_state.get("pred_return", 0.0),
                    last_price=tool_state.get("last_price", 100.0),
                )
                tool_state.update(risk)
        except Exception as exc:
            logger.error("_node_tool failed: %s", exc)
            tool_state = {}
            state["error"] = str(exc)

        state["tool_state"] = tool_state
        return state

    def _node_write(self, state: CopilotState) -> CopilotState:
        try:
            note = self._writer.write_note(
                tool_state=state.get("tool_state", {}),
                retrieval=state.get("retrieval", {}),
                ticker=state["ticker"],
            )
        except Exception as exc:
            logger.error("_node_write failed: %s", exc)
            note = f"[Note generation failed: {exc}]"
            state["error"] = str(exc)

        state["note"] = note
        return state

    def _node_route(self, state: CopilotState) -> CopilotState:
        tool_state = state.get("tool_state", {})
        signal = tool_state.get("signal", "HOLD")
        position_value = (
            state.get("_position_value", 0.0)  # type: ignore[typeddict-item]
            or tool_state.get("position_value", 0.0)
        )
        signal_id = self._make_signal_id(state["ticker"])
        strategy_name = "copilot_default"

        try:
            hitl_result = self._hitl.route(
                signal_id=signal_id,
                ticker=state["ticker"],
                signal=signal,
                position_value=position_value,
                strategy_name=strategy_name,
            )
        except Exception as exc:
            logger.error("_node_route failed: %s", exc)
            hitl_result = {}
            state["error"] = str(exc)

        state["hitl_result"] = hitl_result
        return state

    # ── Sequential path ────────────────────────────────────────────────────────

    def _run_sequential(
        self,
        ticker: str,
        query: str | None,
        position_value: float,
    ) -> dict[str, Any]:
        """Run the pipeline without LangGraph — plain sequential calls."""
        state: CopilotState = {
            "ticker": ticker,
            "query": query,
            "tool_state": {},
            "retrieval": {},
            "note": "",
            "hitl_result": {},
            "error": None,
        }
        state["_position_value"] = position_value  # type: ignore[typeddict-unknown-key]

        state = self._node_retrieve(state)
        state = self._node_tool(state)
        state = self._node_write(state)
        state = self._node_route(state)

        return self._state_to_output(state)

    # ── Output formatting ─────────────────────────────────────────────────────

    @staticmethod
    def _state_to_output(state: CopilotState) -> dict[str, Any]:
        docs = state.get("retrieval", {}).get("documents", [])
        return {
            "tool_state": state.get("tool_state", {}),
            "retrieval": state.get("retrieval", {"documents": []}),
            "note": state.get("note", ""),
            "hitl_result": state.get("hitl_result", {}),
            "sources_used": len(docs),
            "error": state.get("error"),
        }
