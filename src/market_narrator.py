"""Two-agent Market Narrator: ticker-scoped retrieval followed by grounded synthesis."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, TypedDict

from agent_traces import trace_agent_run

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_DIR = Path(os.getenv("MARKET_NARRATOR_TRANSCRIPTS_DIR", REPO_ROOT / "data" / "earnings_transcripts"))
VECTOR_STORE_DIR = Path(os.getenv("MARKET_NARRATOR_VECTOR_STORE", REPO_ROOT / "data" / "market_narrator" / "chroma"))
COLLECTION_NAME = "market_narrator"


class NarrativeState(TypedDict, total=False):
    ticker: str
    prediction: dict[str, Any]
    query: str
    sources: list[dict[str, Any]]
    narrative: str
    run_id: str


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY to enable Market Narrator embeddings and synthesis. "
            "An OpenAI-compatible OPENAI_API_BASE may also be configured."
        )


def _published_at(article: dict[str, Any]) -> str:
    content = article.get("content") if isinstance(article.get("content"), dict) else article
    value = content.get("pubDate") or content.get("providerPublishTime") or ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            return ""
    return str(value)


def _news_fields(article: dict[str, Any]) -> tuple[str, str, str, str]:
    """Normalize the old and current Yahoo Finance news response formats."""
    content = article.get("content") if isinstance(article.get("content"), dict) else article
    title = str(content.get("title") or "Untitled financial news")
    summary = str(content.get("summary") or content.get("description") or "")
    canonical_url = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
    if isinstance(canonical_url, dict):
        url = str(canonical_url.get("url") or content.get("link") or "")
    else:
        url = str(canonical_url or content.get("link") or "")
    provider = content.get("provider") or article.get("publisher") or "Yahoo Finance"
    if isinstance(provider, dict):
        provider = provider.get("displayName") or provider.get("name") or "Yahoo Finance"
    return title, summary, url, str(provider)


def _get_source_documents(ticker: str) -> list[Any]:
    """Load recent Yahoo Finance news and local earnings-call transcript files."""
    from langchain_core.documents import Document

    documents = []
    try:
        import yfinance as yf

        articles = yf.Ticker(ticker).news or []
        for article in articles[:30]:
            title, summary, url, provider = _news_fields(article)
            text = "\n".join(part for part in (title, summary) if part.strip())
            if not text.strip():
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "ticker": ticker,
                    "source_type": "financial_news",
                    "title": title[:300],
                    "source": provider,
                    "url": url,
                    "published_at": _published_at(article),
                },
            ))
    except Exception as exc:
        logger.warning("Yahoo Finance news retrieval failed for %s: %s", ticker, exc)

    ticker_dir = TRANSCRIPTS_DIR / ticker
    if ticker_dir.exists():
        for path in sorted(ticker_dir.glob("**/*")):
            if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                logger.warning("Could not read transcript %s: %s", path, exc)
                continue
            if not text:
                continue
            documents.append(Document(
                page_content=text,
                metadata={
                    "ticker": ticker,
                    "source_type": "earnings_call_transcript",
                    "title": path.stem.replace("_", " "),
                    "source": path.name,
                    "url": "",
                    "published_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).date().isoformat(),
                },
            ))
    return documents


def _new_vector_store():
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    embedding_model = os.getenv("MARKET_NARRATOR_EMBEDDING_MODEL", "text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=OpenAIEmbeddings(model=embedding_model),
    )


def index_ticker_sources(ticker: str) -> int:
    """Refresh a ticker's Chroma documents from recent news and local transcripts."""
    _require_openai_key()
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    ticker = ticker.strip().upper()
    documents = _get_source_documents(ticker)
    if not documents:
        raise ValueError(
            f"No sources found for {ticker}. Yahoo Finance returned no news and no transcript "
            f"files were found under {TRANSCRIPTS_DIR / ticker}. Upload a recent earnings-call "
            "transcript or retry when news is available."
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    store = _new_vector_store()
    try:
        store.delete(where={"ticker": ticker})
    except Exception:
        # A newly-created/empty Chroma collection has nothing to remove.
        logger.debug("No previous Chroma documents to replace for %s", ticker, exc_info=True)
    ids = []
    for index, chunk in enumerate(chunks):
        identity = "|".join((ticker, str(chunk.metadata.get("source")), str(index), chunk.page_content))
        ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def _retrieve_sources(ticker: str, query: str, k: int = 6) -> list[dict[str, Any]]:
    store = _new_vector_store()
    docs = store.similarity_search(query, k=k, filter={"ticker": ticker})
    sources = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        sources.append({
            "citation_id": f"S{index}",
            "text": doc.page_content,
            "title": metadata.get("title") or metadata.get("source") or "Source",
            "source": metadata.get("source") or "Unknown source",
            "source_type": metadata.get("source_type") or "financial_news",
            "published_at": metadata.get("published_at") or "",
            "url": metadata.get("url") or "",
        })
    return sources


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content).strip()


def _build_market_narrator_graph():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph

    def retriever_agent(state: NarrativeState) -> dict[str, Any]:
        ticker = state["ticker"].strip().upper()
        query = state.get("query") or (
            f"{ticker} recent earnings-call commentary, business performance, guidance, "
            "risks, and financial news relevant to near-term investor sentiment"
        )
        with trace_agent_run(
            "retriever_agent", ticker,
            {"query": query, "source_refresh": "Yahoo Finance news and local transcripts"},
            run_id=state.get("run_id"),
        ) as trace:
            indexed_chunks = index_ticker_sources(ticker)
            sources = _retrieve_sources(ticker, query)
            trace.set_output(sources, indexed_chunks=indexed_chunks, retrieved_count=len(sources))
            return {"sources": sources, "query": query}

    def synthesis_agent(state: NarrativeState) -> dict[str, Any]:
        ticker = state["ticker"].strip().upper()
        prediction = state.get("prediction", {})
        sources = state.get("sources", [])
        with trace_agent_run(
            "synthesis_agent", ticker,
            {"prediction": prediction, "retrieved_sources": sources},
            run_id=state.get("run_id"),
        ) as trace:
            model_name = os.getenv("MARKET_NARRATOR_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, temperature=0)
            source_text = "\n\n".join(
                f"[{s['citation_id']}] {s['title']} ({s['source']}; {s['published_at']})\n{s['text']}"
                for s in sources
            ) or "No retrieved source material is available."
            system_prompt = (
                "You are the AI Market Narrator for a stock-return model. Explain the model's "
                "direction in plain English using only the supplied prediction and retrieved "
                "sources. The sources provide context, not proof of what caused the model output. "
                "Never claim news or transcripts were model features unless explicitly stated. "
                "Distinguish model signal from source-based interpretation. Cite each factual "
                "source-based statement with the exact bracket citation ID, e.g. [S1]. Do not "
                "invent citations, facts, dates, or causal links. If evidence is thin, conflicting, "
                "or absent, say so clearly. Explain whether the conformal interval crosses zero "
                "and what that means for uncertainty. End with a brief non-personalized disclaimer: "
                "this is model commentary, not investment advice. Use a concise headline and a few "
                "short paragraphs or bullets; no unsupported price target."
            )
            user_prompt = (
                f"Ticker: {ticker}\nLatest model prediction and conformal interval (JSON):\n"
                f"{prediction}\n\nRetrieved source excerpts (cite only these IDs):\n{source_text}\n\n"
                "Write the bullish/bearish explanation. If point return is near zero or the interval "
                "spans positive and negative returns, describe the signal as mixed/uncertain rather "
                "than forcing a directional label."
            )
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            narrative = _content_to_text(response.content)
            trace.set_output(narrative, model=model_name, citations=[s["citation_id"] for s in sources])
            return {"narrative": narrative}

    graph = StateGraph(NarrativeState)
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("synthesis_agent", synthesis_agent)
    graph.add_edge(START, "retriever_agent")
    graph.add_edge("retriever_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)
    return graph.compile()


def generate_market_narrative(
    ticker: str,
    prediction: dict[str, Any],
    *,
    query: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run retrieval and synthesis and return narrative, retrieved citations, and run id."""
    _require_openai_key()
    ticker = ticker.strip().upper()
    run_id = run_id or __import__("uuid").uuid4().hex
    result = _build_market_narrator_graph().invoke({
        "ticker": ticker,
        "prediction": prediction,
        "query": query or (
            f"{ticker} recent earnings-call commentary and financial news explaining "
            "business drivers, guidance, risks, and sentiment relevant to investors"
        ),
        "run_id": run_id,
    })
    return {
        "ticker": ticker,
        "run_id": run_id,
        "prediction": prediction,
        "narrative": result.get("narrative", ""),
        "sources": result.get("sources", []),
        "query": result.get("query", query or ""),
    }
