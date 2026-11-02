"""Catalog-grounded conversational search implemented as a LangGraph RAG flow."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from src.observability import span


class SearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(max_length=1200)
    selected_ids: list[str] = Field(max_length=12)


class SearchState(TypedDict, total=False):
    query: str
    history: list[dict[str, str]]
    results: list[dict[str, Any]]
    answer: str
    selected_ids: list[str]
    error: str | None


@lru_cache(maxsize=2)
def _sentence_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class SearchRecommendationAgent:
    """Retrieve catalog evidence, then produce a constrained title-ID answer.

    The only titles returned to callers come from retrieved catalog records. The
    model may select IDs from that set; unknown IDs are discarded after Guardrails
    and Pydantic validation.
    """

    def __init__(self, *, qdrant_client: Any = None, embedding_model: str | None = None,
                 collection: str | None = None, llm: Any = None):
        self.collection = collection or os.getenv("CATALOG_COLLECTION", "netflix_catalog")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.client = qdrant_client
        self.llm = llm
        self._graph = None
        try:
            from langgraph.graph import END, START, StateGraph
            graph = StateGraph(SearchState)
            graph.add_node("retrieve", self._retrieve_node)
            graph.add_node("generate", self._generate_node)
            graph.add_edge(START, "retrieve")
            graph.add_edge("retrieve", "generate")
            graph.add_edge("generate", END)
            self._graph = graph.compile()
        except ImportError as exc:
            raise RuntimeError("Install langgraph to enable conversational catalog search") from exc

    def _qdrant(self):
        if self.client is None:
            from qdrant_client import QdrantClient
            url = os.getenv("QDRANT_URL")
            if not url:
                raise RuntimeError("QDRANT_URL is required for catalog search")
            self.client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=2.0)
        return self.client

    def _retrieve_node(self, state: SearchState) -> dict[str, Any]:
        prior = " ".join(message["content"] for message in state.get("history", [])[-4:]
                          if message.get("role") == "user")
        retrieval_query = f"{prior} {state['query']}".strip()
        vector = _sentence_encoder(self.embedding_model).encode(retrieval_query, normalize_embeddings=True).tolist()
        max_runtime = re.search(r"(?:under|less than|shorter than)\s+(\d{1,3})\s*(?:minutes|min)\b", state["query"], re.I)
        query_filter = None
        if max_runtime:
            from qdrant_client import models
            query_filter = models.Filter(must=[models.FieldCondition(
                key="duration_seconds", range=models.Range(lte=int(max_runtime.group(1)) * 60))])
        with span("catalog.vector_retrieval"):
            result = self._qdrant().query_points(collection_name=self.collection, query=vector,
                        query_filter=query_filter, limit=40, with_payload=True, score_threshold=None)
        titles = []
        for point in result.points:
            payload = dict(point.payload or {})
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            title = str(payload.get("title") or meta.get("title") or payload.get("name") or "").strip()
            # Upload IDs are assets; only catalog entries with a display title can ground answers.
            if not title:
                continue
            item = {**meta, **payload, "title": title, "id": str(payload.get("content_id", point.id)),
                    "score": float(point.score)}
            item.pop("vector", None)
            titles.append(item)
        return {"results": titles[:12], "error": None}

    def _get_llm(self):
        if self.llm is None:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=os.getenv("RECOMMENDATION_LLM_MODEL", "gpt-4o-mini"),
                                  temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        return self.llm

    def _generate_node(self, state: SearchState) -> dict[str, Any]:
        sources = state.get("results", [])
        if not sources:
            return {"answer": "I couldn't find a matching title in the available catalog.", "selected_ids": []}
        source_ids = {item["id"] for item in sources}
        try:
            structured = self._get_llm().with_structured_output(SearchAnswer)
            response = structured.invoke([
                ("system", "You are a streaming catalog search assistant. Recommend only titles from the supplied catalog evidence. Return selected_ids from that evidence; never invent titles, availability, or facts. If none fit, return an empty selected_ids list. Keep the answer concise."),
                ("user", json.dumps({"request": state["query"], "conversation_history": state.get("history", [])[-8:], "catalog_evidence": [
                    {"id": item["id"], "title": item["title"], "description": item.get("description") or item.get("text", ""),
                     "genres": item.get("genres", []), "duration_seconds": item.get("duration_seconds"),
                     "year": item.get("year")} for item in sources]}, ensure_ascii=False)),
            ])
            output = response.model_dump_json() if isinstance(response, BaseModel) else json.dumps(response)
            # Guardrails validates the structured JSON against the same Pydantic output contract.
            try:
                from guardrails import Guard
                guard = Guard.for_pydantic(output_class=SearchAnswer)
                outcome = guard.validate(output)
                guarded = outcome.validated_output
                answer = guarded if isinstance(guarded, SearchAnswer) else SearchAnswer.model_validate(guarded)
            except ImportError as exc:
                raise RuntimeError("Install guardrails-ai to validate conversational search output") from exc
            valid_ids = [item_id for item_id in answer.selected_ids if item_id in source_ids]
            valid_ids = list(dict.fromkeys(valid_ids))
            names_by_id = {item["id"]: item["title"] for item in sources}
            grounded_names = [names_by_id[item_id] for item_id in valid_ids]
            safe_answer = ("Catalog matches: " + ", ".join(grounded_names) + ".") if grounded_names else "I couldn't find a matching title in the available catalog."
            return {"answer": safe_answer, "selected_ids": valid_ids}
        except Exception:
            # Catalog evidence still yields a safe, useful answer when the LLM is unavailable.
            selected = sources[:5]
            return {"answer": "Here are the closest matches in the catalog.",
                    "selected_ids": [item["id"] for item in selected]}

    def search(self, query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        if not query.strip() or len(query) > 1000:
            raise ValueError("query must contain 1 to 1000 characters")
        history = history or []
        state = self._graph.invoke({"query": query.strip(), "history": history[-12:],
                                    "results": [], "selected_ids": [], "answer": ""})
        by_id = {item["id"]: item for item in state.get("results", [])}
        return {"query": query, "answer": state.get("answer", ""),
                "recommendations": [by_id[item_id] for item_id in state.get("selected_ids", []) if item_id in by_id]}


@lru_cache(maxsize=1)
def get_search_agent() -> SearchRecommendationAgent:
    """Reuse the LangGraph and Qdrant client between API requests."""
    return SearchRecommendationAgent()
