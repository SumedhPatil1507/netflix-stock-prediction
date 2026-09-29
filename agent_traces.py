"""
Agent trace logger proxy for root-level accessibility.
Redirects to src.agent_traces for unified observability.
"""
from src.agent_traces import (
    LangfuseAgentTracer,
    get_agent_tracer,
    log_agent_run,
    get_recent_traces,
    get_trace_summary,
    TRACES_PATH,
)

__all__ = [
    "LangfuseAgentTracer",
    "get_agent_tracer",
    "log_agent_run",
    "get_recent_traces",
    "get_trace_summary",
    "TRACES_PATH",
]
