"""Strict, shared explainable recommendation response contract."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_NEXT_STEPS = [
    "Step 1: Prefetch the 1080p HLS manifest through the edge CDN when available.",
    "Step 2: Log the impression event to the Kafka interaction queue.",
    "Step 3: Update the online user profile feature vector in Redis.",
]


class ExplainableRecommendation(BaseModel):
    """Exact API envelope; extra output properties are rejected."""

    model_config = ConfigDict(extra="forbid")
    verdict: str = Field(min_length=1, max_length=500)
    reasoning: str = Field(min_length=1, max_length=3000)
    recommendation: str = Field(min_length=1, max_length=3000)
    next_steps: list[str] = Field(min_length=1, max_length=8)


def response(*, verdict: str, cosine_similarity: float | None, rationale: str,
             recommendation: str, next_steps: list[str] | None = None) -> dict[str, Any]:
    """Build and validate the contract, including the vector similarity evidence."""
    similarity = (f"Cosine vector similarity: {cosine_similarity:.3f} "
                  f"({cosine_similarity * 100:.1f}%).") if cosine_similarity is not None else "Cosine vector similarity: unavailable for this event."
    return ExplainableRecommendation(
        verdict=verdict[:500],
        reasoning=f"{similarity} {rationale}"[:3000],
        recommendation=recommendation[:3000],
        next_steps=next_steps or DEFAULT_NEXT_STEPS,
    ).model_dump()
