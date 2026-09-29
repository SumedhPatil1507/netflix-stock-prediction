"""
Tests for AI Market Narrator (Multi-Agent RAG, ChromaDB Vector Store, Langfuse Tracing, RAGAS Eval).
"""
import pytest
import os
import json
from src.narrator.vector_store import get_market_vector_store
from src.narrator.retriever_agent import RetrieverAgent, run_retriever_agent
from src.narrator.synthesis_agent import SynthesisAgent
from src.narrator.graph import run_market_narrator
from src.narrator.eval import evaluate_narrative_faithfulness
from src.agent_traces import get_agent_tracer, log_agent_run, get_recent_traces


def test_vector_store_initialization_and_retrieval():
    """Verify vector store seeds corpus and returns relevant documents."""
    store = get_market_vector_store()
    assert len(store.index.documents) > 0
    results = store.search("revenue subscriber growth", ticker="NFLX", n_results=3)
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]
    assert "score" in results[0]


def test_retriever_agent():
    """Verify retriever agent handles ticker and custom queries."""
    agent = RetrieverAgent()
    data = agent.retrieve(ticker="NFLX", query="operating margin", n_results=4)
    assert "citations" in data
    assert "formatted_context" in data
    assert "documents" in data
    assert len(data["citations"]) >= 1


def test_synthesis_agent_output_structure():
    """Verify synthesis agent produces structured narrative with citations and model bounds."""
    agent = SynthesisAgent()
    model_state = agent.fetch_model_state("NFLX")
    retriever = RetrieverAgent()
    retrieved = retriever.retrieve(ticker="NFLX", n_results=3)
    
    res = agent.synthesize(
        ticker="NFLX",
        retrieved_context=retrieved["formatted_context"],
        citations=retrieved["citations"],
        model_state=model_state
    )
    assert "narrative" in res
    assert "citations" in res
    assert len(res["narrative"]) > 50
    assert len(res["citations"]) > 0


def test_ragas_evaluation():
    """Verify RAGAS evaluation calculates composite metric between 0 and 1."""
    narrative = "According to [1], Netflix recorded strong subscriber growth with operating margin of 28%. The stance is bullish."
    retrieved_docs = [
        {"id": "doc_1", "text": "Netflix recorded strong subscriber growth with operating margin of 28%.", "score": 0.9}
    ]
    citations = [{"citation_id": "[1]", "title": "Q4 Earnings"}]
    model_prediction = {"signal": "BULLISH", "predicted_return_pct": 1.5}
    scores = evaluate_narrative_faithfulness(narrative, retrieved_docs, citations, model_prediction)
    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "citation_grounding" in scores
    assert "ragas_composite_score" in scores
    assert 0.0 <= scores["ragas_composite_score"] <= 1.0


def test_agent_traces_logging():
    """Verify structured trace logging and retrieval."""
    tracer = get_agent_tracer()
    trace = log_agent_run(
        ticker="NFLX",
        retrieval_query="test query",
        retrieved_docs=[{"id": "doc_test", "metadata": {"title": "Test Doc", "source_type": "news"}, "score": 0.9}],
        model_prediction={"predicted_return_pct": 0.85, "signal": "BUY"},
        narrative="Test synthesis narrative with [1].",
        citations=[{"citation_id": "[1]", "title": "Test Doc"}],
        ragas_scores={"faithfulness": 0.95, "ragas_composite_score": 0.92},
        latency_ms=12.5,
    )
    assert "trace_id" in trace
    assert trace["status"] == "SUCCESS"
    recent = get_recent_traces(limit=5)
    assert len(recent) > 0
    assert any(t.get("trace_id") == trace["trace_id"] for t in recent)


def test_end_to_end_narrator_graph():
    """Verify full LangGraph pipeline execution."""
    result = run_market_narrator(ticker="NFLX", custom_query="monetization")
    assert "narrative" in result
    assert "citations" in result
    assert "ragas_scores" in result
    assert "model_state" in result
    assert "trace_id" in result
    assert result["model_state"]["ticker"] == "NFLX"
