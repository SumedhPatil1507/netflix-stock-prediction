"""Context-aware second-stage multi-task candidate ranker."""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ["retrieval_score", "recency", "popularity", "runtime", "session_duration",
            "device_code", "hour_sin", "hour_cos", "weekend", "genre_abandonment",
            "continue_progress", "genre_affinity"]
POSITIVE = {"play", "pause", "thumbs_up"}
NEGATIVE = {"thumbs_down", "abandon"}


def _date(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _genres(title: dict[str, Any]) -> set[str]:
    value = title.get("genres", []) or []
    return {str(x).casefold() for x in (value.split(",") if isinstance(value, str) else value)}


class ContextualRanker:
    """XGBoost relevance plus completion prediction, with safe cold-start scoring."""

    def __init__(self):
        self.relevance_model = None
        self.completion_model = None
        self.is_trained = False

    def feature_row(self, title: dict[str, Any], *, user_events: list[dict[str, Any]], context: dict[str, Any],
                    retrieval_score: float = 0.0) -> list[float]:
        now = _date(context.get("now") or datetime.now(timezone.utc).isoformat())
        item_id = str(title.get("content_id", ""))
        interactions = [e for e in user_events if str(e.get("content_id")) == item_id]
        def recency_weight(event: dict[str, Any]) -> float:
            age_seconds = max(0.0, (now - _date(event.get("occurred_at"))).total_seconds())
            return math.exp(-age_seconds / (14 * 86400))

        duration = max(1.0, float(title.get("duration_seconds") or 0.0))
        progress = max((float(e.get("position_seconds") or 0.0) / duration for e in interactions), default=0.0)
        progress = min(1.0, max(0.0, progress))
        device = str(context.get("device_type", "unknown")).casefold()
        device_code = {"tv": 1.0, "mobile": 0.5, "tablet": 0.65, "desktop": 0.8}.get(device, 0.0)
        hour = float(context.get("hour", now.hour)) + now.minute / 60
        item_genres = _genres(title)
        matching_recent = [recency_weight(event) for event in user_events
                           if event.get("event_type") in POSITIVE and item_genres.intersection(_genres(event))]
        release_recency = 0.0
        release_value = title.get("release_date") or title.get("release_year") or title.get("year")
        if release_value:
            try:
                value = str(release_value)
                release_date = (datetime(int(value), 1, 1, tzinfo=timezone.utc)
                                if re.fullmatch(r"\d{4}", value)
                                else datetime.fromisoformat(value.replace("Z", "+00:00")))
                if release_date.tzinfo is None:
                    release_date = release_date.replace(tzinfo=timezone.utc)
                release_recency = math.exp(-max(0.0, (now - release_date).total_seconds()) / (365 * 86400))
            except (TypeError, ValueError):
                release_recency = 0.0
        own_recency = max((recency_weight(event) for event in interactions), default=0.0)
        recency = max(matching_recent, default=0.0, )
        recency = max(recency, release_recency, own_recency)
        abandon_weight: dict[str, float] = {}
        positive_weight: dict[str, float] = {}
        for event in user_events:
            target = abandon_weight if event.get("event_type") == "abandon" else positive_weight if event.get("event_type") in POSITIVE else None
            if target is not None:
                for genre in _genres(event):
                    target[genre] = target.get(genre, 0.0) + recency_weight(event)
        genre_abandonment = (sum(abandon_weight.get(g, 0.0) for g in item_genres) /
                             (len(item_genres) + sum(abandon_weight.values()))) if item_genres else 0.0
        genre_affinity = (sum(positive_weight.get(g, 0.0) for g in item_genres) /
                          (len(item_genres) + sum(positive_weight.values()))) if item_genres else 0.0
        try:
            session_duration = max(0.0, float(context.get("session_duration_seconds", 0)))
        except (TypeError, ValueError):
            session_duration = 0.0
        return [float(retrieval_score), recency, float(title.get("popularity", 0) or 0),
                min(duration / 3600, 5.0), min(session_duration / 3600, 5.0), device_code,
                math.sin(hour * math.tau / 24), math.cos(hour * math.tau / 24),
                float(context.get("weekend", now.weekday() >= 4)), genre_abandonment, progress, genre_affinity]

    def fit(self, events: list[dict[str, Any]], titles: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Fit relevance and completion objectives from contextual interaction rows."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install xgboost to train the contextual ranker") from exc
        rows, labels, completion = [], [], []
        by_user: dict[str, list[tuple[int, datetime, dict[str, Any]]]] = {}
        for index, interaction in enumerate(events):
            catalog_title = titles.get(str(interaction.get("content_id", "")))
            if catalog_title:
                by_user.setdefault(str(interaction.get("user_id", "")), []).append((
                    index, _date(interaction.get("occurred_at")),
                    {**interaction, "genres": catalog_title.get("genres", []) or []}))
        for index, event in enumerate(events):
            title = titles.get(str(event.get("content_id", "")))
            if not title:
                continue
            event_type = str(event.get("event_type", ""))
            if event_type not in POSITIVE | NEGATIVE:
                continue
            context = {key: event.get(key) for key in ("occurred_at", "device_type", "session_duration_seconds")}
            event_time = _date(event.get("occurred_at"))
            history = [item for prior_index, prior_time, item in by_user.get(str(event.get("user_id", "")), [])
                       if prior_index != index and (prior_time < event_time or (prior_time == event_time and prior_index < index))]
            rows.append(self.feature_row(title, user_events=history, context=context))
            labels.append(int(event_type in POSITIVE))
            duration = max(1.0, float(title.get("duration_seconds") or 0))
            completion.append(min(1.0, max(0.0, float(event.get("position_seconds") or 0) / duration)))
        if len(rows) < 4 or len(set(labels)) < 2:
            raise ValueError("Ranker training requires at least four positive/negative contextual events")
        weights = np.ones(len(labels), dtype=np.float32)
        self.relevance_model = XGBClassifier(n_estimators=160, max_depth=5, learning_rate=0.06,
              subsample=0.85, colsample_bytree=0.85, reg_lambda=2, eval_metric="logloss",
              n_jobs=2, random_state=17)
        self.relevance_model.fit(np.asarray(rows), np.asarray(labels), sample_weight=weights)
        self.completion_model = XGBRegressor(n_estimators=120, max_depth=4, learning_rate=0.05,
              subsample=0.85, colsample_bytree=0.85, reg_lambda=2, n_jobs=2, random_state=17)
        self.completion_model.fit(np.asarray(rows), np.asarray(completion))
        self.is_trained = True
        return {"events": len(rows), "positive": sum(labels), "negative": len(labels) - sum(labels)}

    def score(self, title: dict[str, Any], *, user_events: list[dict[str, Any]], context: dict[str, Any],
              retrieval_score: float = 0.0) -> tuple[float, float]:
        features = np.asarray([self.feature_row(title, user_events=user_events, context=context,
                                                retrieval_score=retrieval_score)], dtype=np.float32)
        if self.is_trained:
            relevance = float(self.relevance_model.predict_proba(features)[0][1])
            completion = float(np.clip(self.completion_model.predict(features)[0], 0.0, 1.0))
            return 0.75 * relevance + 0.25 * completion, completion
        recency, popularity, session_duration, _, _, _, _, weekend, abandon, progress, affinity = features[0][1:]
        score = (0.62 * retrieval_score + 0.16 * float(popularity > 0) + 0.12 * affinity +
                 0.10 * recency + 0.08 * weekend - 0.25 * abandon - 0.05 * progress)
        return float(score), float(progress)

    def save(self, path: str | Path) -> None:
        import joblib
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"relevance": self.relevance_model, "completion": self.completion_model}, target)

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, path: str | Path) -> "ContextualRanker":
        import joblib
        result = cls()
        data = joblib.load(path)
        result.relevance_model, result.completion_model = data["relevance"], data["completion"]
        result.is_trained = True
        return result
