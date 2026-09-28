import json
from types import SimpleNamespace

import pytest

import agent_traces
from src import model_registry
from src.market_narrator import _news_fields
from scripts.evaluate_narrative_faithfulness import load_synthesis_runs


def test_latest_prediction_is_stored_per_ticker(monkeypatch, tmp_path):
    prediction_path = tmp_path / "latest_predictions.json"
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": [], "latest": "model_NFLX_latest.pkl"}))
    monkeypatch.setattr(model_registry, "PREDICTIONS_PATH", str(prediction_path))
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", str(registry_path))

    netflix = model_registry.record_latest_prediction(
        {"predicted_return_pct": 1.25, "confidence_interval": {"lower_return_pct": -0.5}}, "nflx"
    )
    apple = model_registry.record_latest_prediction({"predicted_return_pct": -0.2}, "AAPL")

    assert netflix["ticker"] == "NFLX"
    assert netflix["model_version"] == "model_NFLX_latest.pkl"
    assert model_registry.get_latest_prediction("NFLX")["predicted_return_pct"] == 1.25
    assert model_registry.get_latest_prediction("AAPL")["predicted_return_pct"] == -0.2
    assert apple["ticker"] == "AAPL"


def test_news_fields_support_current_yahoo_news_shape():
    title, summary, url, provider = _news_fields({
        "content": {
            "title": "Netflix reports results",
            "summary": "Revenue increased year over year.",
            "canonicalUrl": {"url": "https://example.com/nflx"},
            "provider": {"displayName": "Example Finance"},
            "pubDate": "2026-09-01T12:00:00Z",
        }
    })
    assert title == "Netflix reports results"
    assert summary == "Revenue increased year over year."
    assert url == "https://example.com/nflx"
    assert provider == "Example Finance"


def test_news_fields_support_legacy_yahoo_news_shape():
    title, summary, url, provider = _news_fields({
        "title": "Legacy headline", "link": "https://example.com/legacy", "publisher": "Wire"
    })
    assert title == "Legacy headline"
    assert summary == ""
    assert url == "https://example.com/legacy"
    assert provider == "Wire"


def test_agent_trace_writes_success_and_error_records(monkeypatch, tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(agent_traces, "TRACE_PATH", trace_path)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with agent_traces.trace_agent_run("retriever_agent", "NFLX", {"query": "latest news"}) as trace:
        trace.set_output([{"citation_id": "S1"}], retrieved_count=1)
    with pytest.raises(RuntimeError):
        with agent_traces.trace_agent_run("synthesis_agent", "NFLX", {"sources": []}):
            raise RuntimeError("model unavailable")

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["status"] for record in records] == ["success", "error"]
    assert records[0]["output"][0]["citation_id"] == "S1"
    assert records[0]["metadata"]["retrieved_count"] == 1
    assert "model unavailable" in records[1]["error"]


def test_langgraph_retrieves_then_synthesizes_with_citations(monkeypatch, tmp_path):
    import langchain_openai
    import src.market_narrator as market_narrator

    monkeypatch.setattr(agent_traces, "TRACE_PATH", tmp_path / "graph-traces.jsonl")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(market_narrator, "index_ticker_sources", lambda ticker: 2)
    monkeypatch.setattr(market_narrator, "_retrieve_sources", lambda ticker, query: [{
        "citation_id": "S1", "text": "Revenue grew 12%.", "title": "Quarterly results",
        "source": "Example News", "source_type": "financial_news",
        "published_at": "2026-09-01", "url": "https://example.test/results",
    }])

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        def invoke(self, messages):
            return SimpleNamespace(content="The model is bullish; recent revenue growth supports context [S1].")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    result = market_narrator._build_market_narrator_graph().invoke({
        "ticker": "NFLX",
        "prediction": {"predicted_return_pct": 0.8, "confidence_interval": {"lower_return_pct": -0.2}},
        "query": "latest Netflix results",
        "run_id": "test-run-1",
    })

    assert result["sources"][0]["citation_id"] == "S1"
    assert "[S1]" in result["narrative"]
    records = [json.loads(line) for line in (tmp_path / "graph-traces.jsonl").read_text().splitlines()]
    assert [record["agent"] for record in records] == ["retriever_agent", "synthesis_agent"]
    assert all(record["run_id"] == "test-run-1" for record in records)


def test_agent_trace_sends_span_to_langfuse_when_configured(monkeypatch, tmp_path):
    import langfuse

    class FakeObservation:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    observation = FakeObservation()

    class FakeContext:
        def __enter__(self):
            return observation

        def __exit__(self, *args):
            return False

    class FakeClient:
        def __init__(self):
            self.flushed = False

        def start_as_current_observation(self, **kwargs):
            assert kwargs["as_type"] == "span"
            assert kwargs["name"] == "market-narrator.retriever_agent"
            return FakeContext()

        def flush(self):
            self.flushed = True

    client = FakeClient()
    monkeypatch.setattr(langfuse, "get_client", lambda: client)
    monkeypatch.setattr(agent_traces, "TRACE_PATH", tmp_path / "langfuse-traces.jsonl")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")

    with agent_traces.trace_agent_run("retriever_agent", "NFLX", {"query": "results"}) as trace:
        trace.set_output({"retrieved_count": 1})

    assert client.flushed is True
    assert len(observation.updates) == 2
    assert observation.updates[0]["input"] == {"query": "results"}
    assert observation.updates[1]["output"] == {"retrieved_count": 1}


def test_ragas_loader_reads_synthesis_output_and_retrieved_context(tmp_path):
    trace_file = tmp_path / "agent_traces.jsonl"
    trace_file.write_text(json.dumps({
        "event": "agent_run",
        "agent": "synthesis_agent",
        "status": "success",
        "run_id": "eval-1",
        "ticker": "NFLX",
        "inputs": {"retrieved_sources": [{"text": "Revenue increased 12%."}]},
        "output": "Revenue increased 12% [S1].",
        "finished_at": "2026-09-01T12:00:00Z",
    }) + "\n", encoding="utf-8")

    samples = load_synthesis_runs(trace_file)
    assert len(samples) == 1
    assert samples[0]["run_id"] == "eval-1"
    assert samples[0]["retrieved_contexts"] == ["Revenue increased 12%."]
    assert samples[0]["response"] == "Revenue increased 12% [S1]."
