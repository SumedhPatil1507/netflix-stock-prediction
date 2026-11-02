"""Offline recommendation training and Qdrant indexing CLI.

Input files are JSON or JSONL arrays containing catalog records and interaction
events. See README for the required fields and training order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ranking import ContextualRanker
from .two_tower import TwoTowerModel


def _load(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON array or one JSON object per line")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the two-tower model and contextual ranker")
    parser.add_argument("--titles", required=True, help="JSON/JSONL catalog with content_id, title, genres, duration_seconds")
    parser.add_argument("--events", required=True, help="JSON/JSONL implicit interaction events")
    parser.add_argument("--artifact-dir", default="models/recommendations")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()
    titles, events = _load(args.titles), _load(args.events)
    model = TwoTowerModel()
    print("Two tower:", model.fit(titles, events, epochs=args.epochs))
    artifact_dir = Path(args.artifact_dir)
    model.save(artifact_dir / "two_tower.json")
    try:
        ranker = ContextualRanker()
        print("Contextual ranker:", ranker.fit(events, {str(t["content_id"]): t for t in titles}))
        ranker.save(artifact_dir / "contextual_ranker.joblib")
    except ValueError as exc:
        print(f"Contextual ranker not fitted: {exc}")
    if not args.skip_index:
        print("Qdrant title vectors indexed:", model.index_qdrant(titles))


if __name__ == "__main__":
    main()
