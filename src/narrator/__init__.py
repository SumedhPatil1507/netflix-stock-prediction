"""
AI Market Narrator — Agentic RAG and Synthesis Layer.

Components:
- Retriever Agent: Vector search over ChromaDB containing earnings calls & financial news.
- Synthesis Agent: Integrates model predictions, conformal bounds, and retrieved evidence into citations-backed analysis.
- LangGraph Pipeline: Multi-agent orchestration state graph.
- Langfuse Observability: End-to-end trace, latency, and token monitoring.
- RAGAS Evaluator: Automated faithfulness and context scoring.
"""
from src.narrator.graph import run_market_narrator, compile_narrator_graph
from src.narrator.vector_store import get_market_vector_store
from src.narrator.eval import evaluate_narrative_faithfulness

__all__ = [
    "run_market_narrator",
    "compile_narrator_graph",
    "get_market_vector_store",
    "evaluate_narrative_faithfulness",
]
