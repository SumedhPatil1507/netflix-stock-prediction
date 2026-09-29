"""
Observability and Trace Logging with Langfuse.
Records agent execution traces, spans, generation metrics, and RAG retrieval evaluations.
Supports live Langfuse cloud credentials and robust local structured JSONL trace logging.
"""
from __future__ import annotations
import os
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
TRACES_PATH = os.path.join(LOGS_DIR, "agent_traces.jsonl")


class LangfuseAgentTracer:
    """
    Langfuse observability manager.
    Coordinates cloud trace shipping and local trace persistence.
    """
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.langfuse_client = None
        self._init_client()
        os.makedirs(LOGS_DIR, exist_ok=True)

    def _init_client(self):
        if self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                logger.info("Langfuse client connected successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}. Falling back to local tracing.")
                self.langfuse_client = None
        else:
            logger.info("Langfuse credentials not detected. Operating in local structured tracing mode.")

    def trace_agent_run(
        self,
        ticker: str,
        retrieval_query: str,
        retrieved_docs: List[Dict[str, Any]],
        model_prediction: Dict[str, Any],
        narrative: str,
        citations: List[Dict[str, Any]],
        ragas_scores: Optional[Dict[str, float]] = None,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a complete agent pipeline run trace.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        trace_payload = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "ticker": ticker.upper(),
            "status": "SUCCESS",
            "latency_ms": round(latency_ms, 2),
            "spans": {
                "retriever_agent": {
                    "query": retrieval_query,
                    "docs_retrieved": len(retrieved_docs),
                    "sources": [
                        {
                            "id": d.get("id"),
                            "title": d.get("metadata", {}).get("title"),
                            "source_type": d.get("metadata", {}).get("source_type"),
                            "score": d.get("score"),
                        } for d in retrieved_docs
                    ],
                },
                "synthesis_agent": {
                    "predicted_return_pct": model_prediction.get("predicted_return_pct"),
                    "signal": model_prediction.get("signal"),
                    "confidence_interval": model_prediction.get("confidence_interval"),
                    "narrative_length_chars": len(narrative),
                    "citations_count": len(citations),
                },
                "ragas_evaluation": ragas_scores or {},
            },
            "output": {
                "narrative_preview": narrative[:200] + "..." if len(narrative) > 200 else narrative,
                "citations": citations,
            },
            "metadata": metadata or {"pipeline": "LangGraph_AI_Market_Narrator", "version": "2.0.0"}
        }

        # 1. Ship to Langfuse Cloud if configured
        if self.langfuse_client:
            try:
                lf_trace = self.langfuse_client.trace(
                    name="ai_market_narrator",
                    id=trace_id,
                    metadata={"ticker": ticker, **(metadata or {})},
                    tags=["langgraph", "rag", ticker.lower()]
                )
                
                # Retriever span
                lf_trace.span(
                    name="retriever_agent",
                    input={"query": retrieval_query, "ticker": ticker},
                    output={"retrieved_count": len(retrieved_docs), "docs": retrieved_docs[:3]}
                )

                # Synthesis generation
                lf_trace.generation(
                    name="synthesis_agent",
                    input={"model_prediction": model_prediction, "context_count": len(retrieved_docs)},
                    output=narrative,
                    metadata={"citations": citations}
                )

                # Ragas scores logging
                if ragas_scores:
                    for metric_name, score_val in ragas_scores.items():
                        lf_trace.score(name=f"ragas_{metric_name}", value=score_val)

                self.langfuse_client.flush()
                trace_payload["langfuse_synced"] = True
            except Exception as e:
                logger.warning(f"Error shipping trace to Langfuse: {e}")
                trace_payload["langfuse_synced"] = False
        else:
            trace_payload["langfuse_synced"] = False

        # 2. Append to local JSONL trace log
        try:
            with open(TRACES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to write local trace: {e}")

        return trace_payload


_GLOBAL_TRACER: Optional[LangfuseAgentTracer] = None

def get_agent_tracer() -> LangfuseAgentTracer:
    """Singleton getter for Langfuse tracer."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = LangfuseAgentTracer()
    return _GLOBAL_TRACER


def log_agent_run(
    ticker: str,
    retrieval_query: str,
    retrieved_docs: List[Dict[str, Any]],
    model_prediction: Dict[str, Any],
    narrative: str,
    citations: List[Dict[str, Any]],
    ragas_scores: Optional[Dict[str, float]] = None,
    latency_ms: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Top-level functional API for trace logging."""
    tracer = get_agent_tracer()
    return tracer.trace_agent_run(
        ticker=ticker,
        retrieval_query=retrieval_query,
        retrieved_docs=retrieved_docs,
        model_prediction=model_prediction,
        narrative=narrative,
        citations=citations,
        ragas_scores=ragas_scores,
        latency_ms=latency_ms,
        metadata=metadata,
    )


def get_recent_traces(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve recent traces from the local trace log."""
    if not os.path.exists(TRACES_PATH):
        return []
    traces = []
    try:
        with open(TRACES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"Error reading traces: {e}")
    return traces[-limit:][::-1]


def get_trace_summary() -> Dict[str, Any]:
    """Summarize trace statistics for dashboards."""
    traces = get_recent_traces(limit=100)
    if not traces:
        return {
            "total_runs": 0,
            "avg_latency_ms": 0.0,
            "latest_trace_id": None,
            "langfuse_configured": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
        }
    
    latencies = [t.get("latency_ms", 0.0) for t in traces if t.get("latency_ms")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "total_runs": len(traces),
        "avg_latency_ms": round(avg_lat, 2),
        "latest_trace_id": traces[0].get("trace_id"),
        "langfuse_configured": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
        "recent_traces": traces[:5],
    }
