#!/usr/bin/env python3
"""Score recorded Market Narrator responses against the context retrieved for each run.

Usage:
    OPENAI_API_KEY=... python scripts/evaluate_narrative_faithfulness.py
    python scripts/evaluate_narrative_faithfulness.py --trace-file logs/agent_traces.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_FILE = ROOT / "logs" / "agent_traces.jsonl"
DEFAULT_OUTPUT_FILE = ROOT / "outputs" / "narrative_faithfulness.json"


def load_synthesis_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}. Generate a narrative first.")
    runs = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed trace line {line_number}", file=sys.stderr)
                continue
            if record.get("event") != "agent_run" or record.get("agent") != "synthesis_agent":
                continue
            inputs = record.get("inputs") or {}
            sources = inputs.get("retrieved_sources") or []
            response = record.get("output")
            if record.get("status") != "success" or not response:
                continue
            contexts = [str(source.get("text", "")).strip() for source in sources]
            contexts = [context for context in contexts if context]
            if not contexts:
                continue
            runs.append({
                "run_id": record.get("run_id"),
                "ticker": record.get("ticker"),
                "user_input": (
                    f"Explain the latest model direction for {record.get('ticker', 'the ticker')} "
                    "using retrieved earnings-call and financial-news context."
                ),
                "response": str(response),
                "retrieved_contexts": contexts,
                "generated_at": record.get("finished_at"),
            })
    if not runs:
        raise ValueError("No successful synthesis traces with retrieved source context were found.")
    return runs


async def evaluate(runs: list[dict[str, Any]], model: str):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    client = AsyncOpenAI(base_url=os.getenv("OPENAI_API_BASE") or None)
    evaluator_llm = llm_factory(model, client=client)
    faithfulness = Faithfulness(llm=evaluator_llm)
    results = []
    for run in runs:
        result = await faithfulness.ascore(
            user_input=run["user_input"],
            response=run["response"],
            retrieved_contexts=run["retrieved_contexts"],
        )
        results.append({
            "run_id": run["run_id"],
            "ticker": run["ticker"],
            "faithfulness": float(result.value),
            "context_count": len(run["retrieved_contexts"]),
            "generated_at": run["generated_at"],
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY must be set to run RAGAS faithfulness evaluation")
    try:
        runs = load_synthesis_runs(args.trace_file)
        scores = asyncio.run(evaluate(runs, args.model))
    except ImportError as exc:
        parser.error(f"Install evaluation dependencies with `pip install -r requirements-dev.txt`: {exc}")
    summary = {
        "metric": "ragas_faithfulness",
        "model": args.model,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "run_count": len(scores),
        "mean_faithfulness": sum(row["faithfulness"] for row in scores) / len(scores),
        "scores": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
