"""Implicit-feedback two-tower candidate retrieval model backed by Qdrant."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

POSITIVE = {"play", "pause", "thumbs_up"}
NEGATIVE = {"thumbs_down", "abandon"}


def title_text(title: dict[str, Any]) -> str:
    genres = title.get("genres", [])
    if isinstance(genres, str):
        genres = [genres]
    return " ".join(str(value) for value in (
        title.get("title", ""), title.get("description", ""), title.get("synopsis", ""),
        title.get("language", ""), " ".join(map(str, genres)),
    ) if value)


class TwoTowerModel:
    """Train user-ID and title-text towers with BPR on implicit interactions.

    Qdrant stores projected title vectors; normalized user vectors are queried
    against that ANN index. Training is offline and artifacts are CPU portable.
    """

    def __init__(self, *, embedding_model: str | None = None, dimension: int = 128):
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = dimension
        self.title_ids: list[str] = []
        self.user_ids: list[str] = []
        self.title_vectors = np.empty((0, dimension), dtype=np.float32)
        self.semantic_vectors = np.empty((0, 0), dtype=np.float32)
        self.user_vectors = np.empty((0, dimension), dtype=np.float32)
        self.title_index: dict[str, int] = {}
        self.user_index: dict[str, int] = {}
        self.seen: dict[str, set[str]] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-catalog.txt for title embeddings") from exc
        encoder = SentenceTransformer(self.embedding_model)
        return np.asarray(encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def fit(self, titles: list[dict[str, Any]], events: list[dict[str, Any]], *, epochs: int = 8,
            learning_rate: float = 0.025, seed: int = 17) -> dict[str, Any]:
        """Fit user and title towers; negative samples include explicit dislikes."""
        import torch
        from torch import nn
        from torch.nn import functional as F

        if not titles:
            raise ValueError("At least one catalog title is required")
        by_id = {str(item["content_id"]): item for item in titles if item.get("content_id")}
        self.title_ids = sorted(by_id)
        if not self.title_ids:
            raise ValueError("Catalog title records must include content_id")
        title_index = {content_id: i for i, content_id in enumerate(self.title_ids)}
        base = self._encode([title_text(by_id[item_id]) for item_id in self.title_ids])

        positive_pairs: list[tuple[int, int, float]] = []
        dislikes: dict[int, list[int]] = {}
        raw_seen: dict[str, set[str]] = {}
        users = sorted({str(e.get("user_id")) for e in events if e.get("user_id")})
        user_index = {user_id: i for i, user_id in enumerate(users)}
        for event in events:
            user_id, content_id = str(event.get("user_id", "")), str(event.get("content_id", ""))
            if user_id not in user_index or content_id not in title_index:
                continue
            raw_seen.setdefault(user_id, set()).add(content_id)
            typ = str(event.get("event_type", ""))
            pair = (user_index[user_id], title_index[content_id])
            if typ in POSITIVE:
                weight = 1.5 if typ == "thumbs_up" else 0.65 if typ == "pause" else 1.0
                positive_pairs.append((*pair, weight))
            elif typ in NEGATIVE:
                dislikes.setdefault(pair[0], []).append(pair[1])
        if not users or not positive_pairs:
            raise ValueError("Training needs users and at least one positive play/pause/thumbs_up event")

        torch.manual_seed(seed)
        user_tower = nn.Embedding(len(users), self.dimension)
        projection = nn.Linear(base.shape[1], self.dimension, bias=False)
        nn.init.normal_(user_tower.weight, std=0.08)
        optimizer = torch.optim.Adam([*user_tower.parameters(), *projection.parameters()], lr=learning_rate)
        base_tensor = torch.from_numpy(base)
        pairs = np.asarray(positive_pairs, dtype=np.float32)
        rng = np.random.default_rng(seed)
        popularity = np.bincount(pairs[:, 1].astype(int), minlength=len(self.title_ids)).astype(np.float32)
        negative_prob = np.power(popularity + 1, 0.75)
        negative_prob /= negative_prob.sum()

        for _ in range(max(1, epochs)):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), 2048):
                batch = pairs[start:start + 2048]
                user_ids_batch = torch.as_tensor(batch[:, 0].astype(np.int64))
                pos_ids = torch.as_tensor(batch[:, 1].astype(np.int64))
                neg_ids_np = rng.choice(len(self.title_ids), size=len(batch), p=negative_prob)
                for row, user_idx in enumerate(user_ids_batch.tolist()):
                    explicit = dislikes.get(user_idx, [])
                    if explicit and rng.random() < 0.6:
                        neg_ids_np[row] = rng.choice(explicit)
                    else:
                        user_name = users[user_idx]
                        seen_ids = raw_seen.get(user_name, set())
                        for _attempt in range(16):
                            sampled = int(rng.choice(len(self.title_ids), p=negative_prob))
                            if self.title_ids[sampled] not in seen_ids:
                                neg_ids_np[row] = sampled
                                break
                        else:
                            if len(self.title_ids) > 1 and neg_ids_np[row] == batch[row, 1]:
                                neg_ids_np[row] = (int(neg_ids_np[row]) + 1) % len(self.title_ids)
                neg_ids = torch.as_tensor(neg_ids_np.astype(np.int64))
                pos_items = F.normalize(projection(base_tensor[pos_ids]), dim=1)
                neg_items = F.normalize(projection(base_tensor[neg_ids]), dim=1)
                user_vectors = F.normalize(user_tower.weight, dim=1)
                pos_score = (user_vectors[user_ids_batch] * pos_items).sum(dim=1)
                neg_score = (user_vectors[user_ids_batch] * neg_items).sum(dim=1)
                weights = torch.as_tensor(batch[:, 2])
                loss = (F.softplus(neg_score - pos_score) * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.user_ids = users
        with torch.no_grad():
            item_vectors = F.normalize(projection(base_tensor), dim=1).cpu().numpy()
            user_vectors = F.normalize(user_tower.weight, dim=1).cpu().numpy()
        self.title_vectors = item_vectors
        self.semantic_vectors = base
        self.user_vectors = user_vectors
        self.title_index = title_index
        self.user_index = user_index
        self.seen = raw_seen
        return {"users": len(users), "titles": len(by_id), "positive_events": len(positive_pairs),
                "explicit_negative_users": len(dislikes), "embedding_dimension": self.dimension}

    def user_vector(self, user_id: str) -> list[float] | None:
        """Return trained tower output or an interaction-profile vector for a new user."""
        if user_id in self.user_index:
            return self.user_vectors[self.user_index[user_id]].tolist()
        seen = self.seen.get(user_id, set())
        vectors = [self.title_vectors[self.title_index[item]] for item in seen if item in self.title_index]
        if not vectors:
            return None
        vector = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        norm = np.linalg.norm(vector) or 1.0
        return (vector / norm).tolist()

    def title_vector(self, content_id: str) -> list[float] | None:
        index = self.title_index.get(str(content_id))
        return self.title_vectors[index].tolist() if index is not None else None

    def index_qdrant(self, titles: list[dict[str, Any]], *, url: str | None = None,
                     api_key: str | None = None, collection: str | None = None) -> int:
        """Upsert projected title vectors and catalog payload into Qdrant."""
        from qdrant_client import QdrantClient, models
        import uuid

        endpoint = url or os.getenv("QDRANT_URL")
        if not endpoint:
            raise RuntimeError("QDRANT_URL is required for candidate retrieval")
        name = collection or os.getenv("RECOMMENDATION_COLLECTION", "netflix_recommendations")
        client = QdrantClient(url=endpoint, api_key=api_key or os.getenv("QDRANT_API_KEY"))
        dim = int(self.title_vectors.shape[1])
        if not client.collection_exists(name):
            client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        _ensure_payload_index(client, name, "content_id", models.PayloadSchemaType.KEYWORD)
        points = []
        for title in titles:
            content_id = str(title.get("content_id", ""))
            vector_index = self.title_index.get(content_id)
            if vector_index is None:
                continue
            points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, content_id)), vector=self.title_vectors[vector_index].tolist(),
                                             payload={**title, "content_id": content_id}))
        if points:
            client.upsert(collection_name=name, points=points, wait=True)
        # Keep a second, text-embedding collection for natural-language RAG queries.
        semantic_name = os.getenv("CATALOG_COLLECTION", "netflix_catalog")
        semantic_dim = int(self.semantic_vectors.shape[1])
        if not client.collection_exists(semantic_name):
            client.create_collection(semantic_name, vectors_config=models.VectorParams(
                size=semantic_dim, distance=models.Distance.COSINE))
        _ensure_payload_index(client, semantic_name, "duration_seconds", models.PayloadSchemaType.FLOAT)
        semantic_points = []
        for title in titles:
            content_id = str(title.get("content_id", ""))
            vector_index = self.title_index.get(content_id)
            if vector_index is not None:
                semantic_points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, content_id)),
                          vector=self.semantic_vectors[vector_index].tolist(), payload={**title, "content_id": content_id,
                          "text": title_text(title)}))
        if semantic_points:
            client.upsert(collection_name=semantic_name, points=semantic_points, wait=True)
        return len(points)


    def save(self, path: str | Path) -> None:
        import numpy as np
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target.with_suffix(".npz"), title_vectors=self.title_vectors, user_vectors=self.user_vectors)
        target.write_text(json.dumps({"embedding_model": self.embedding_model, "dimension": self.dimension,
                         "title_ids": self.title_ids, "user_ids": self.user_ids,
                         "seen": {key: sorted(value) for key, value in self.seen.items()}}, separators=(",", ":")),
                          encoding="utf-8")

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, path: str | Path) -> "TwoTowerModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(embedding_model=data["embedding_model"], dimension=data["dimension"])
        model.title_ids, model.user_ids = data["title_ids"], data["user_ids"]
        with np.load(Path(path).with_suffix(".npz")) as vectors:
            model.title_vectors = vectors["title_vectors"]
            model.user_vectors = vectors["user_vectors"]
        model.title_index = {item_id: index for index, item_id in enumerate(model.title_ids)}
        model.user_index = {user_id: index for index, user_id in enumerate(model.user_ids)}
        model.seen = {key: set(value) for key, value in data.get("seen", {}).items()}
        return model


def _ensure_payload_index(client: Any, collection: str, field_name: str, field_schema: Any) -> None:
    schema = getattr(client.get_collection(collection), "payload_schema", {}) or {}
    if field_name not in schema:
        from qdrant_client import models
        client.create_payload_index(collection_name=collection, field_name=field_name,
                                    field_schema=field_schema, wait=True)
