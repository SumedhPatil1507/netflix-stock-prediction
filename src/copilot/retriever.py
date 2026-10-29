"""
CopilotRetriever — RAG retrieval layer for the Research Copilot.

Wraps narrator's VectorStore and CorpusManager (no code duplication).
Ingests and retrieves earnings transcripts, 10-K/10-Q excerpts, and
recent news for a given ticker.
"""
from __future__ import annotations

import logging
from typing import Any

from src.narrator.vector_store import VectorStore
from src.narrator.corpus import CorpusManager

logger = logging.getLogger(__name__)


class CopilotRetriever:
    """
    Retrieval component of the Research Copilot.

    Delegates to :class:`src.narrator.vector_store.VectorStore` and
    :class:`src.narrator.corpus.CorpusManager` — no duplication of
    embedding or corpus-management logic.

    Parameters
    ----------
    ticker : Default ticker symbol used when none is passed to methods.
    collection_name : ChromaDB collection name.
    persist_dir : Directory where ChromaDB persists its data.
    """

    def __init__(
        self,
        ticker: str = "NFLX",
        collection_name: str = "financial_documents",
        persist_dir: str = "data/chroma_db",
    ) -> None:
        self.default_ticker = ticker
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        self.corpus_manager = CorpusManager()
        logger.info(
            "CopilotRetriever initialised (collection=%s, ticker=%s)",
            collection_name,
            ticker,
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_ticker(self, ticker: str) -> None:
        """
        Load sample transcripts and news for *ticker* and add them to the
        vector store.  Safe to call multiple times (ChromaDB deduplicates by ID).

        Parameters
        ----------
        ticker : Stock symbol to ingest documents for.
        """
        try:
            documents, metadatas, ids = (
                self.corpus_manager.prepare_documents_for_vector_store(ticker)
            )
            if documents:
                self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(
                    "Ingested %d documents for ticker %s", len(documents), ticker
                )
            else:
                logger.warning("No documents prepared for ticker %s", ticker)
        except Exception as exc:
            logger.error("ingest_ticker failed for %s: %s", ticker, exc)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Retrieve the *n_results* most relevant documents for *query*.

        Delegates to :meth:`VectorStore.query` with an optional ticker
        metadata filter.  Falls back gracefully to a filter-free query if
        the ticker filter matches no documents.

        Parameters
        ----------
        query     : Free-text retrieval query.
        ticker    : Ticker to filter results to (defaults to *self.default_ticker*).
        n_results : Number of results to return.

        Returns
        -------
        dict with keys:
            ``query``, ``ticker``, ``documents`` (list[dict]), ``timestamp``.
        """
        from datetime import datetime

        _ticker = ticker or self.default_ticker
        try:
            raw = self.vector_store.query(
                query_text=query,
                n_results=n_results,
                where={"ticker": _ticker},
            )
            docs = _flatten_results(raw)

            # If no docs matched the ticker filter, try without filter
            if not docs:
                logger.debug(
                    "Ticker filter returned 0 docs for %s; retrying without filter.",
                    _ticker,
                )
                raw = self.vector_store.query(
                    query_text=query,
                    n_results=n_results,
                )
                docs = _flatten_results(raw)

            logger.info(
                "CopilotRetriever.retrieve: %d docs for '%s' (ticker=%s)",
                len(docs),
                query[:60],
                _ticker,
            )
        except Exception as exc:
            logger.error("retrieve failed: %s", exc)
            docs = []

        return {
            "query": query,
            "ticker": _ticker,
            "documents": docs,
            "timestamp": datetime.now().isoformat(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert ChromaDB query result dict into a flat list of dicts."""
    ids = raw.get("ids", [[]])[0]
    texts = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    docs: list[dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        docs.append(
            {
                "id": doc_id,
                "text": texts[i] if i < len(texts) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return docs
