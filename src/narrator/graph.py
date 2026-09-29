"""
LangGraph Multi-Agent Pipeline for AI Market Narrator.
Coordinates the Retriever Agent, Synthesis Agent, RAGAS Evaluator, and Langfuse Tracer.
"""
from __future__ import annotations
import time
import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END

from src.narrator.retriever_agent import RetrieverAgent
from src.narrator.synthesis_agent import SynthesisAgent
from src.narrator.eval import evaluate_narrative_faithfulness
from src.agent_traces import log_agent_run

logger = logging.getLogger(__name__)


class NarratorState(TypedDict, total=False):
    ticker: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    formatted_context: str
    model_state: Dict[str, Any]
    narrative: str
    ragas_scores: Dict[str, float]
    trace_id: str
    latency_ms: float
    status: str


# ── Node 1: Retriever Agent Node ──────────────────────────────────────────────
def retriever_node(state: NarratorState) -> NarratorState:
    """Retrieve financial transcripts and market news from ChromaDB."""
    ticker = state.get("ticker", "NFLX")
    query = state.get("query")
    agent = RetrieverAgent()
    
    retrieval_res = agent.retrieve(ticker=ticker, query=query, n_results=4)
    
    return {
        **state,
        "query": retrieval_res["query"],
        "retrieved_docs": retrieval_res["documents"],
        "citations": retrieval_res["citations"],
        "formatted_context": retrieval_res["formatted_context"],
    }


# ── Node 2: Synthesis Agent Node ──────────────────────────────────────────────
def synthesis_node(state: NarratorState) -> NarratorState:
    """Synthesize model registry predictions with retrieved context."""
    ticker = state.get("ticker", "NFLX")
    formatted_context = state.get("formatted_context", "")
    citations = state.get("citations", [])
    model_state = state.get("model_state")
    
    agent = SynthesisAgent()
    if model_state is None:
        model_state = agent.fetch_model_state(ticker)
        
    synthesis_res = agent.synthesize(
        ticker=ticker,
        retrieved_context=formatted_context,
        citations=citations,
        model_state=model_state
    )
    
    return {
        **state,
        "model_state": synthesis_res["model_state"],
        "narrative": synthesis_res["narrative"],
        "citations": synthesis_res["citations"],
    }


# ── Node 3: Evaluation & Observability Node ───────────────────────────────────
def eval_and_trace_node(state: NarratorState) -> NarratorState:
    """Run RAGAS evaluation and log trace with Langfuse."""
    ticker = state.get("ticker", "NFLX")
    narrative = state.get("narrative", "")
    retrieved_docs = state.get("retrieved_docs", [])
    citations = state.get("citations", [])
    model_state = state.get("model_state", {})
    start_time = state.get("_start_time", time.time())
    latency_ms = (time.time() - start_time) * 1000.0

    # RAGAS faithfulness & relevance evaluation
    ragas_scores = evaluate_narrative_faithfulness(
        narrative=narrative,
        retrieved_docs=retrieved_docs,
        citations=citations,
        model_prediction=model_state,
    )

    # Langfuse Trace Logging
    trace_record = log_agent_run(
        ticker=ticker,
        retrieval_query=state.get("query", ""),
        retrieved_docs=retrieved_docs,
        model_prediction=model_state,
        narrative=narrative,
        citations=citations,
        ragas_scores=ragas_scores,
        latency_ms=latency_ms,
        metadata={"pipeline": "LangGraph_AI_Market_Narrator", "version": "2.0.0"}
    )

    return {
        **state,
        "ragas_scores": ragas_scores,
        "trace_id": trace_record.get("trace_id", ""),
        "latency_ms": round(latency_ms, 2),
        "status": "COMPLETED",
    }


# ── LangGraph Pipeline Construction ───────────────────────────────────────────
def compile_narrator_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(NarratorState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("eval_and_trace", eval_and_trace_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "synthesis")
    workflow.add_edge("synthesis", "eval_and_trace")
    workflow.add_edge("eval_and_trace", END)

    return workflow.compile()


_COMPILED_GRAPH = None

def get_narrator_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = compile_narrator_graph()
    return _COMPILED_GRAPH


def run_market_narrator(
    ticker: str = "NFLX",
    custom_query: Optional[str] = None,
    model_prediction: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the end-to-end AI Market Narrator pipeline via LangGraph.
    """
    graph = get_narrator_graph()
    
    initial_state: NarratorState = {
        "ticker": ticker.upper(),
        "query": custom_query or "",
        "_start_time": time.time(),
    }
    if model_prediction is not None:
        initial_state["model_state"] = model_prediction

    result = graph.invoke(initial_state)
    return result
