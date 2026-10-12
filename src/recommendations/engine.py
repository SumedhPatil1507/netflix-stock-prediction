"""Two-stage Qdrant ANN retrieval and contextual recommendation rails."""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from .ranking import ContextualRanker, POSITIVE, _genres
from .two_tower import TwoTowerModel
from .explain import response as explain_response
from src.observability import candidate_retrieval_seconds, ranking_seconds, observe, span


class QdrantCandidateGenerator:
    """Stage 1: retrieve up to 500 personalized title candidates with cosine ANN."""

    def __init__(self, *, url: str | None = None, api_key: str | None = None,
                 collection: str | None = None, client: Any = None):
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Install qdrant-client to serve recommendations") from exc
            endpoint = url or os.getenv("QDRANT_URL")
            if not endpoint:
                raise RuntimeError("QDRANT_URL is required")
            client = QdrantClient(url=endpoint, api_key=api_key or os.getenv("QDRANT_API_KEY"), timeout=2.0)
        self.client = client
        self.collection = collection or os.getenv("RECOMMENDATION_COLLECTION", "netflix_recommendations")

    def retrieve(self, vector: list[float], *, limit: int = 500, exclude_ids: list[str] | None = None) -> list[dict[str, Any]]:
        from qdrant_client import models
        must_not = []
        if exclude_ids:
            must_not.append(models.FieldCondition(key="content_id", match=models.MatchAny(any=exclude_ids)))
        result = self.client.query_points(collection_name=self.collection, query=vector,
                    query_filter=models.Filter(must_not=must_not) if must_not else None,
                    limit=min(500, max(1, limit)), with_payload=True, score_threshold=None)
        candidates = []
        for point in result.points:
            payload = dict(point.payload or {})
            candidates.append({**payload, "content_id": str(payload.get("content_id", point.id)),
                              "retrieval_score": float(point.score)})
        return candidates

    def by_ids(self, content_ids: list[str]) -> list[dict[str, Any]]:
        if not content_ids:
            return []
        from qdrant_client import models
        result = self.client.scroll(collection_name=self.collection,
            scroll_filter=models.Filter(must=[models.FieldCondition(key="content_id", match=models.MatchAny(any=content_ids))]),
            limit=len(content_ids), with_payload=True, with_vectors=False)
        return [dict(point.payload or {}) for point in result[0]]


@lru_cache(maxsize=1)
def default_candidate_generator() -> QdrantCandidateGenerator:
    """Keep one Qdrant client's HTTP connection pool per API process."""
    return QdrantCandidateGenerator()


class RecommendationEngine:
    """Stage 1 retrieval followed by stage 2 multi-task contextual ranking."""

    def __init__(self, two_tower: TwoTowerModel, ranker: ContextualRanker | None = None,
                 candidates: QdrantCandidateGenerator | None = None):
        self.two_tower = two_tower
        self.ranker = ranker or ContextualRanker()
        self.candidates = candidates or default_candidate_generator()

    def recommend(self, user_id: str, *, events: list[dict[str, Any]], context: dict[str, Any] | None = None,
                  limit: int = 500) -> dict[str, Any]:
        context = context or {}
        vector = self.two_tower.user_vector(user_id)
        if vector is None:
            # Cold start from available positive event title vectors.
            positives = [str(e.get("content_id")) for e in events if e.get("event_type") in POSITIVE]
            vectors = [vector for item in positives if (vector := self.two_tower.title_vector(item)) is not None]
            if vectors:
                import numpy as np
                mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
                vector = (mean / (np.linalg.norm(mean) or 1.0)).tolist()
        if vector is None:
            raise ValueError("No interaction profile is available for this user")

        seen_event_types = POSITIVE | {"seek", "thumbs_down", "abandon"}
        seen = sorted({str(e.get("content_id")) for e in events if e.get("event_type") in seen_event_types})
        retrieval_started = time.perf_counter()
        with span("recommendation.vector_retrieval"):
            candidates = self.candidates.retrieve(vector, limit=limit, exclude_ids=seen)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        observe(candidate_retrieval_seconds, retrieval_ms / 1000)
        title_map = {str(item.get("content_id")): item for item in candidates}
        history_ids = sorted({str(event.get("content_id")) for event in events if event.get("content_id")})
        title_map.update({str(item.get("content_id")): item for item in self.candidates.by_ids(history_ids)})
        # Attach genres to event records so genre-level abandon/affinity is measurable.
        for event in events:
            metadata = title_map.get(str(event.get("content_id")))
            if metadata:
                event.setdefault("genres", metadata.get("genres", []))

        ranking_started = time.perf_counter()
        ranked = []
        with span("recommendation.contextual_ranking"):
            for item in candidates:
                score, completion = self.ranker.score(item, user_events=events, context=context,
                                                      retrieval_score=float(item.get("retrieval_score", 0)))
                ranked.append({**item, "score": score, "predicted_completion": completion})
            ranked.sort(key=lambda item: (item["score"], item.get("retrieval_score", 0)), reverse=True)
        ranking_ms = (time.perf_counter() - ranking_started) * 1000
        observe(ranking_seconds, ranking_ms / 1000)

        # Restore resumable titles independently of retrieval suppression.
        progress_events = []
        for event in events:
            item_id = str(event.get("content_id", ""))
            title = self.two_tower.title_vector(item_id)
            if event.get("event_type") in {"play", "pause"} and title:
                progress = float(event.get("position_seconds") or 0)
                duration = float(event.get("duration_seconds") or
                                 title_map.get(item_id, {}).get("duration_seconds") or 0)
                fraction = progress / duration if duration else 0
                if 0.05 <= fraction < 0.95:
                    progress_events.append((fraction, item_id))
        resume_ids = list(dict.fromkeys(item_id for _, item_id in sorted(progress_events, reverse=True)))
        resume_titles = self.candidates.by_ids(resume_ids)
        for title in resume_titles:
            event = next((e for e in reversed(events) if str(e.get("content_id")) == str(title.get("content_id"))), {})
            duration = max(1.0, float(title.get("duration_seconds") or event.get("duration_seconds") or 0))
            title["continue_progress"] = min(1.0, float(event.get("position_seconds") or 0) / duration)

        positive_genres = {genre for event in events if event.get("event_type") in POSITIVE for genre in _genres(event)}
        because = [item for item in ranked if positive_genres.intersection(_genres(item))]
        hour = int(context.get("hour", 20) or 20)
        is_weekend = bool(context.get("weekend", False))
        weekend = [item for item in ranked if is_weekend and 18 <= hour <= 23]
        rows = [
            {"id": "continue_watching", "title": "Continue Watching", "items": resume_titles[:20]},
            {"id": "top_picks", "title": "Top Picks for Weekend Night", "items": weekend[:20]},
            {"id": "because_watched", "title": "Because You Watched", "items": because[:20]},
            {"id": "for_you", "title": "Top Picks for You", "items": ranked[:20]},
        ]
        primary_row = next((row for row in rows if row["id"] == "for_you" and row["items"]),
                           next((row for row in rows if row["items"]), None))
        if primary_row is None:
            return explain_response(verdict="No matching title", cosine_similarity=None,
                    rationale="No eligible catalog title was returned by candidate retrieval.",
                    recommendation="Refresh the catalog or ask for a different genre, runtime, or mood.")

        top = primary_row["items"][0]
        score = top.get("retrieval_score")
        genre_names = sorted(_genres(top))
        relevant = [event for event in events if set(genre_names).intersection(_genres(event))]
        abandoned = [event for event in relevant if event.get("event_type") == "abandon"]
        if relevant:
            abandon_rate = 100.0 * len(abandoned) / len(relevant)
            abandon_text = f"observed {abandon_rate:.1f}% abandonment across {len(relevant)} matching-genre interactions"
        else:
            abandon_text = "no matching-genre interactions are available to estimate abandonment"
        hour = int(context.get("hour", 20) or 20)
        time_label = "late-night" if hour >= 21 or hour < 5 else "evening" if hour >= 18 else "daytime"
        device = context.get("device_type", "unknown")
        rationale = (f"Context: {time_label} viewing on {device}; selected genres {', '.join(genre_names) or 'not tagged'}; "
                     f"{abandon_text}. Candidate retrieval took {retrieval_ms:.2f} ms and ranking took {ranking_ms:.2f} ms.")
        row_names = [row["title"] for row in rows if row["items"]]
        ordered_titles = [str(item.get("title", "Untitled")) for row in rows for item in row["items"][:3]]
        if top.get("content_type") in {"series", "episode", "tv"}:
            playback = "queue the next episode"
        elif float(top.get("duration_seconds") or 0) <= 1800 and top.get("duration_seconds"):
            playback = "suggest this short title for the current session"
        else:
            playback = "offer its trailer before playback"
        arrangement = (f"Arrange rows in this order: {', '.join(row_names)}. Start with {top.get('title', 'the top title')}; "
                       f"{playback}. Next titles: {', '.join(ordered_titles[:8])}.")
        return explain_response(verdict=f"{primary_row['title']}: {top.get('title', 'Top match')}",
                cosine_similarity=float(score) if score is not None else None,
                rationale=rationale, recommendation=arrangement)
