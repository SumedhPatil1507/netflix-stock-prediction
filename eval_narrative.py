#!/usr/bin/env python
"""
Standalone evaluation script for the AI Market Narrator pipeline.
Runs a full RAG narrative generation for NFLX and scores it with
NarrativeEvaluator (RAGAS if available, otherwise lexical fallback).

Usage:
    python eval_narrative.py
"""
from __future__ import annotations
import os
import sys
import json
import logging

# Resolve project root so relative imports work wherever the script is called from
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # ── 1. Import narrator components ──────────────────────────────────────────
    try:
        from src.narrator import NarratorGraph, VectorStore, CorpusManager
        from src.narrator.eval import NarrativeEvaluator
    except ImportError as e:
        logger.error(f"Could not import narrator components: {e}")
        sys.exit(1)

    # ── 2. Generate narrative ───────────────────────────────────────────────────
    logger.info("Initialising NarratorGraph …")
    narrator = NarratorGraph(
        vector_store=VectorStore(),
        corpus_manager=CorpusManager()
    )

    result = narrator.run(
        ticker="NFLX",
        prediction=0.005,
        conformal_interval=(-0.003, 0.013),
        current_price=650.0
    )

    if not result.get("success"):
        logger.error(f"Narrative generation failed: {result.get('error')}")
        sys.exit(1)

    narrative      = result["narrative"]
    retrieved_docs = result.get("retrieved_documents") or []
    query          = result.get("query") or "NFLX stock analysis bullish factors earnings news"

    logger.info(f"Narrative generated ({len(narrative)} chars), {len(retrieved_docs)} docs retrieved")

    # ── 3. Evaluate ─────────────────────────────────────────────────────────────
    logger.info("Running faithfulness evaluation …")
    evaluator = NarrativeEvaluator()
    eval_result = evaluator.evaluate_narrative(narrative, retrieved_docs, query)

    # ── 4. Print report ─────────────────────────────────────────────────────────
    report = evaluator.generate_evaluation_report(eval_result)
    print(report)

    # ── 5. Save results ──────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    evaluator.save_evaluation_results(eval_result, "outputs/evaluation_results.json")

    # ── 6. Final summary ─────────────────────────────────────────────────────────
    scores = eval_result.get("scores", {})
    faith  = scores.get("faithfulness", "N/A")
    method = "lexical (fallback)" if eval_result.get("fallback") else "RAGAS"
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Ticker           : NFLX")
    print(f"  Sentiment        : {result.get('sentiment', 'N/A')}")
    print(f"  Prediction       : {result['prediction']*100:+.2f}%")
    print(f"  Sources used     : {result.get('sources_used', len(retrieved_docs))}")
    print(f"  Faithfulness     : {faith}")
    print(f"  Evaluation method: {method}")
    print(f"  Results saved to : outputs/evaluation_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
