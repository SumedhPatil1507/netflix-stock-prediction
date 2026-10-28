#!/usr/bin/env python
"""
RAGAS evaluation of the Research Copilot retrieval precision and citation faithfulness.
Run: python eval_ragas_copilot.py
Outputs: outputs/eval_copilot_ragas.json
"""
from __future__ import annotations
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    from src.copilot import CopilotGraph
    from src.narrator.eval import NarrativeEvaluator

    tickers = ["NFLX", "AAPL"]
    evaluator = NarrativeEvaluator()
    results = []

    for t in tickers:
        print(f"Evaluating Copilot for {t}...")
        try:
            g = CopilotGraph(ticker=t)
            r = g.run(ticker=t)
            docs = r.get("retrieval", {}).get("documents", [])
            note = r.get("note", "")
            query = r.get("tool_state", {}).get("query", f"{t} stock analysis")
            eval_r = evaluator.evaluate_narrative(note, docs, query)
            faith = eval_r.get("scores", {}).get("faithfulness", 0.0)
            results.append(
                {
                    "ticker": t,
                    "faithfulness": faith,
                    "sources": len(docs),
                    "fallback": eval_r.get("fallback", True),
                }
            )
            print(f"  {t}: faithfulness={faith:.3f}, sources={len(docs)}")
        except Exception as e:
            print(f"  {t} failed: {e}")
            results.append({"ticker": t, "error": str(e)})

    os.makedirs("outputs", exist_ok=True)
    summary = {
        "avg_faithfulness": sum(r.get("faithfulness", 0) for r in results if "faithfulness" in r)
        / max(len(results), 1)
    }
    payload = {"results": results, "summary": summary}
    with open("outputs/eval_copilot_ragas.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved to outputs/eval_copilot_ragas.json")


if __name__ == "__main__":
    main()
