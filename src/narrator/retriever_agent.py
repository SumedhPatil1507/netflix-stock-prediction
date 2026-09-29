"""
Retriever Agent for Financial RAG.
Formulates multi-vector queries, searches ChromaDB, and structures context with citation markers.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional

from src.narrator.vector_store import get_market_vector_store

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    RAG Retriever Agent specialized in financial earnings calls and market news.
    """
    def __init__(self):
        self.vector_store = get_market_vector_store()

    def formulate_query(self, ticker: str, signal: Optional[str] = None, predicted_return: Optional[float] = None) -> str:
        """
        Formulate a rich query targeting fundamental catalysts and sentiment drivers.
        """
        t = ticker.upper()
        sentiment_intent = "growth operating margin cash flow monetization"
        if signal and "BEAR" in signal.upper():
            sentiment_intent = "margin pressure churn competition currency headwinds risk"
        elif signal and "BULL" in signal.upper():
            sentiment_intent = "operating margin expansion subscriber growth ad-tier revenue live sports"

        return f"{t} earnings call transcript financial news {sentiment_intent} subscriber ARM guidance"

    def retrieve(self, ticker: str, query: Optional[str] = None, n_results: int = 4) -> Dict[str, Any]:
        """
        Query the vector store and return structured retrieved documents with citation tags.
        """
        search_query = query or self.formulate_query(ticker)
        raw_results = self.vector_store.search(query=search_query, ticker=ticker, n_results=n_results)

        citations = []
        context_blocks = []

        for idx, item in enumerate(raw_results, 1):
            meta = item.get("metadata", {})
            title = meta.get("title", f"Document {idx}")
            src_type = meta.get("source_type", "financial_news")
            date_str = meta.get("date", "Recent")
            text = item.get("text", "")
            score = item.get("score", 0.0)

            citation_entry = {
                "citation_id": f"[{idx}]",
                "doc_id": item.get("id"),
                "title": title,
                "source_type": src_type,
                "date": date_str,
                "relevance_score": score,
                "excerpt": text[:180] + "..." if len(text) > 180 else text,
                "full_text": text,
            }
            citations.append(citation_entry)

            type_label = "Earnings Call Transcript" if src_type == "earnings_transcript" else "Financial News"
            context_blocks.append(
                f"Source [{idx}] ({type_label} · {date_str}) — {title}\n"
                f"Content: \"{text}\""
            )

        formatted_context = "\n\n".join(context_blocks)

        return {
            "query": search_query,
            "ticker": ticker.upper(),
            "retrieved_count": len(raw_results),
            "documents": raw_results,
            "citations": citations,
            "formatted_context": formatted_context,
        }


def run_retriever_agent(ticker: str, query: Optional[str] = None, n_results: int = 4) -> Dict[str, Any]:
    """Functional wrapper for RetrieverAgent."""
    agent = RetrieverAgent()
    return agent.retrieve(ticker=ticker, query=query, n_results=n_results)
