"""
Agent traces with Langfuse observability.
Logs every agent run for monitoring and debugging.
"""
from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("Langfuse not installed. Observability features will be disabled.")

logger = logging.getLogger(__name__)

# JSONL fallback log path (relative to project root, resolved at import time)
_JSONL_LOG_PATH = Path(__file__).parent.parent / "logs" / "agent_traces.jsonl"


def _append_jsonl(
    agent_name: str,
    inputs: Any,
    outputs: Any,
    duration_seconds: float,
    ticker: Optional[str] = None,
    error: Optional[str] = None
) -> None:
    """Append one JSONL entry to the fallback trace log. Never raises."""
    try:
        _JSONL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "ticker": ticker,
            "inputs_summary": str(inputs)[:200],
            "outputs_summary": str(outputs)[:200],
            "duration_seconds": round(duration_seconds, 4),
            "error": error,
        }
        with _JSONL_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning(f"JSONL trace write failed: {exc}")


class AgentTracer:
    """Tracer for agent runs using Langfuse observability."""
    
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        enable_tracing: bool = True
    ):
        """
        Initialize agent tracer.
        
        Args:
            public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (defaults to LANGFUSE_HOST env var)
            enable_tracing: Whether to enable tracing
        """
        self.enable_tracing = enable_tracing and LANGFUSE_AVAILABLE
        
        if self.enable_tracing:
            self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
            self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
            self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            
            if self.public_key and self.secret_key:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                logger.info("Langfuse tracer initialized successfully")
            else:
                logger.warning("Langfuse credentials not provided. Tracing disabled.")
                self.enable_tracing = False
        else:
            self.client = None
            logger.info("Agent tracing disabled")
    
    @contextmanager
    def trace_agent_run(
        self,
        agent_name: str,
        inputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing an agent run.
        
        Args:
            agent_name: Name of the agent being traced
            inputs: Input parameters for the agent
            metadata: Additional metadata to log
            
        Yields:
            trace object for updating with results
        """
        start_time = datetime.now()

        if not self.enable_tracing:
            try:
                yield None
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                _append_jsonl(
                    agent_name=agent_name,
                    inputs=inputs,
                    outputs=metadata,
                    duration_seconds=duration,
                    ticker=inputs.get("ticker") if isinstance(inputs, dict) else None,
                )
            return

        trace = None

        try:
            # Create a new trace
            trace = self.client.trace(
                name=agent_name,
                input=inputs,
                metadata=metadata or {}
            )

            logger.info(f"Started trace for agent: {agent_name}")
            yield trace

        except Exception as e:
            logger.error(f"Error creating trace: {e}")
            yield None
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            if trace:
                try:
                    trace.end()
                    logger.info(f"Ended trace for agent: {agent_name} (duration: {duration:.2f}s)")
                except Exception as e:
                    logger.error(f"Error ending trace: {e}")
            _append_jsonl(
                agent_name=agent_name,
                inputs=inputs,
                outputs=metadata,
                duration_seconds=duration,
                ticker=inputs.get("ticker") if isinstance(inputs, dict) else None,
            )
    
    def log_retriever_run(
        self,
        query: str,
        ticker: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a retriever agent run.
        
        Args:
            query: Query used for retrieval
            ticker: Stock ticker symbol
            results: Retrieval results
            metadata: Additional metadata
        """
        try:
            with self.trace_agent_run(
                agent_name="retriever_agent",
                inputs={"query": query, "ticker": ticker},
                metadata=metadata
            ) as trace:
                if trace:
                    trace.update(
                        output={
                            "num_documents": len(results.get("documents", [])),
                            "query": query,
                            "ticker": ticker
                        }
                    )
                    
                    # Log each retrieved document
                    for i, doc in enumerate(results.get("documents", [])):
                        trace.span(
                            name=f"document_{i}",
                            input={"doc_id": doc.get("id")},
                            output={
                                "source": doc.get("metadata", {}).get("source"),
                                "date": doc.get("metadata", {}).get("date"),
                                "relevance": 1 - doc.get("distance", 0) if doc.get("distance") else None
                            }
                        )
            
            _append_jsonl(
                agent_name="retriever_agent",
                inputs={"query": query, "ticker": ticker},
                outputs={"num_documents": len(results.get("documents", []))},
                duration_seconds=0.0,
                ticker=ticker,
            )
            logger.info(f"Logged retriever run for {ticker}")
            
        except Exception as e:
            logger.error(f"Error logging retriever run: {e}")
    
    def log_synthesis_run(
        self,
        prediction: float,
        ticker: str,
        narrative: str,
        citations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a synthesis agent run.
        
        Args:
            prediction: Model prediction
            ticker: Stock ticker symbol
            narrative: Generated narrative
            citations: Citations used in narrative
            metadata: Additional metadata
        """
        try:
            with self.trace_agent_run(
                agent_name="synthesis_agent",
                inputs={
                    "prediction": prediction,
                    "ticker": ticker,
                    "num_citations": len(citations)
                },
                metadata=metadata
            ) as trace:
                if trace:
                    trace.update(
                        output={
                            "narrative_length": len(narrative),
                            "num_citations": len(citations),
                            "sentiment": "bullish" if prediction > 0 else "bearish"
                        }
                    )
                    
                    # Log citations
                    for i, citation in enumerate(citations):
                        trace.span(
                            name=f"citation_{i}",
                            input=citation
                        )
            
            _append_jsonl(
                agent_name="synthesis_agent",
                inputs={"prediction": prediction, "ticker": ticker},
                outputs={"narrative_length": len(narrative), "num_citations": len(citations)},
                duration_seconds=0.0,
                ticker=ticker,
            )
            logger.info(f"Logged synthesis run for {ticker}")
            
        except Exception as e:
            logger.error(f"Error logging synthesis run: {e}")
    
    def log_graph_run(
        self,
        workflow_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a complete graph/workflow run.
        
        Args:
            workflow_name: Name of the workflow
            inputs: Workflow inputs
            outputs: Workflow outputs
            metadata: Additional metadata
        """
        try:
            with self.trace_agent_run(
                agent_name=workflow_name,
                inputs=inputs,
                metadata=metadata
            ) as trace:
                if trace:
                    trace.update(
                        output={
                            "success": outputs.get("success", False),
                            "ticker": outputs.get("ticker"),
                            "sentiment": outputs.get("sentiment"),
                            "narrative_length": len(outputs.get("narrative", "")) if outputs.get("narrative") else 0,
                            "num_citations": len(outputs.get("citations", []))
                        }
                    )
            
            _append_jsonl(
                agent_name=workflow_name,
                inputs=inputs,
                outputs={"success": outputs.get("success"), "sentiment": outputs.get("sentiment")},
                duration_seconds=0.0,
                ticker=outputs.get("ticker") or (inputs.get("ticker") if isinstance(inputs, dict) else None),
            )
            logger.info(f"Logged graph run: {workflow_name}")
            
        except Exception as e:
            logger.error(f"Error logging graph run: {e}")
    
    def log_error(
        self,
        agent_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error that occurred during agent execution.
        
        Args:
            agent_name: Name of the agent
            error: Exception that occurred
            context: Additional context
        """
        if not self.enable_tracing:
            return
        
        try:
            with self.trace_agent_run(
                agent_name=f"{agent_name}_error",
                inputs=context or {},
                metadata={"error_type": type(error).__name__}
            ) as trace:
                if trace:
                    trace.update(
                        output={
                            "error": str(error),
                            "error_type": type(error).__name__
                        }
                    )
            
            logger.error(f"Logged error for agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Error logging error trace: {e}")
    
    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enable_tracing and self.client:
            try:
                self.client.flush()
                logger.info("Flushed traces to Langfuse")
            except Exception as e:
                logger.error(f"Error flushing traces: {e}")


# Global tracer instance
_global_tracer: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer()
    return _global_tracer


def init_tracer(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    enable_tracing: bool = True
) -> AgentTracer:
    """
    Initialize the global tracer instance.
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        host: Langfuse host URL
        enable_tracing: Whether to enable tracing
        
    Returns:
        Initialized tracer instance
    """
    global _global_tracer
    _global_tracer = AgentTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enable_tracing=enable_tracing
    )
    return _global_tracer


# Decorator for automatic tracing
def trace_agent(agent_name: str):
    """
    Decorator to automatically trace agent function calls.
    
    Args:
        agent_name: Name of the agent to trace
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            # Extract inputs from kwargs
            inputs = {}
            if 'ticker' in kwargs:
                inputs['ticker'] = kwargs['ticker']
            if 'query' in kwargs:
                inputs['query'] = kwargs['query']
            if 'prediction' in kwargs:
                inputs['prediction'] = kwargs['prediction']
            
            try:
                with tracer.trace_agent_run(agent_name, inputs):
                    result = func(*args, **kwargs)
                    return result
            except Exception as e:
                tracer.log_error(agent_name, e, {"args": str(args), "kwargs": str(kwargs)})
                raise
        
        return wrapper
    return decorator