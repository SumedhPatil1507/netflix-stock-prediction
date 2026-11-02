"""Deterministic ranking evaluation over 10,000 synthetic interaction sessions."""
from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path


class DeterministicRandom:
    """Small fixed-seed xorshift32 generator for cross-runtime repeatability."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 0x6D2B79F5

    def random(self) -> float:
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state / 4294967296

    def sample(self, population: int, count: int) -> set[int]:
        values = list(range(population))
        for index in range(count):
            selected = index + int(self.random() * (population - index))
            values[index], values[selected] = values[selected], values[index]
        return set(values[:count])


def evaluate(sessions: int = 10_000, k: int = 10, seed: int = 17) -> dict[str, float | int]:
    if sessions < 1 or k < 1:
        raise ValueError("sessions and k must both be positive")
    rng = DeterministicRandom(seed)
    reciprocal_ranks = []
    ndcgs = []
    precisions = []
    catalog_size = 100
    relevant_per_session = 3
    ideal_dcg = sum(1 / log2(rank + 2) for rank in range(min(k, relevant_per_session)))
    for _ in range(sessions):
        relevant = rng.sample(catalog_size, relevant_per_session)
        ranked = sorted(range(catalog_size),
                        key=lambda item: rng.random() + (0.38 if item in relevant else 0.0), reverse=True)
        top_k = ranked[:k]
        hit_ranks = [rank for rank, item in enumerate(top_k, start=1) if item in relevant]
        reciprocal_ranks.append(1 / hit_ranks[0] if hit_ranks else 0.0)
        dcg = sum(1 / log2(rank + 1) for rank, item in enumerate(top_k, start=1) if item in relevant)
        ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
        precisions.append(sum(item in relevant for item in top_k) / k)
    return {
        "sessions": sessions,
        "catalog_size": catalog_size,
        "relevant_titles_per_session": relevant_per_session,
        "k": k,
        "seed": seed,
        "mrr_at_k": sum(reciprocal_ranks) / sessions,
        "ndcg_at_k": sum(ndcgs) / sessions,
        "precision_at_k": sum(precisions) / sessions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.sessions, args.k, args.seed)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
