"""
RAGAS-based Evaluation Script for AI Market Narrator.
Evaluates Faithfulness, Answer Relevancy, Citation Grounding, and Context Precision
against retrieved earnings transcripts and financial news.

Usage:
    python eval_narrative.py --ticker NFLX
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime

# Ensure stdout supports UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.narrator.graph import run_market_narrator


def safe_print(text: str = "") -> None:
    try:
        print(text)
    except Exception:
        try:
            print(text.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass


def run_evaluation(ticker: str = "NFLX", output_path: str = "outputs/narrative_eval.json") -> dict:
    safe_print("======================================================================")
    safe_print(f"  AI Market Narrator - RAGAS Faithfulness & Relevancy Evaluation")
    safe_print(f"  Ticker: {ticker.upper()}  |  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("======================================================================\n")

    safe_print("[*] Executing LangGraph pipeline (Retriever Agent -> Synthesis Agent -> Eval)...")
    res = run_market_narrator(ticker=ticker)

    narrative = res.get("narrative", "")
    citations = res.get("citations", [])
    model_state = res.get("model_state", {})
    scores = res.get("ragas_scores", {})
    trace_id = res.get("trace_id", "N/A")
    latency = res.get("latency_ms", 0.0)

    safe_print(f"[+] Pipeline executed successfully in {latency:.2f} ms")
    safe_print(f"[+] Langfuse Trace ID: {trace_id}\n")

    safe_print("----------------------------------------------------------------------")
    safe_print("                      RAGAS EVALUATION METRICS                        ")
    safe_print("----------------------------------------------------------------------")
    safe_print(f"  - Faithfulness Score:        {scores.get('faithfulness', 0.0):.4f}  (Grounding in retrieved context)")
    safe_print(f"  - Answer Relevancy:          {scores.get('answer_relevancy', 0.0):.4f}  (Alignment with model signal & CP)")
    safe_print(f"  - Citation Grounding:        {scores.get('citation_grounding', 0.0):.4f}  (Valid [Doc #] reference ratio)")
    safe_print(f"  - Context Precision:         {scores.get('context_precision', 0.0):.4f}  (Retrieved corpus signal quality)")
    safe_print("  ------------------------------------------------------------------")
    safe_print(f"  [+] RAGAS Composite Score:   {scores.get('ragas_composite_score', 0.0):.4f} / 1.0000")
    safe_print("----------------------------------------------------------------------\n")

    safe_print(f"[+] Retrieved {len(citations)} Citation Sources:")
    for c in citations:
        safe_print(f"    {c.get('citation_id', '')} {c.get('title', '')} ({c.get('source_type', '')}) - Rel: {c.get('relevance_score', 0):.2f}")

    safe_print("\n----------------------------------------------------------------------")
    safe_print("                   NARRATIVE PREVIEW & SYNTHESIS                      ")
    safe_print("----------------------------------------------------------------------")
    safe_print(narrative[:600] + ("..." if len(narrative) > 600 else ""))
    safe_print("----------------------------------------------------------------------\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_data = {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_id,
        "latency_ms": latency,
        "ragas_scores": scores,
        "model_state": model_state,
        "citations_count": len(citations),
        "citations": citations,
        "narrative": narrative,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    safe_print(f"[+] Full evaluation report saved to: {output_path}")
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Market Narrator with RAGAS metrics.")
    parser.add_argument("--ticker", type=str, default="NFLX", help="Ticker symbol (e.g. NFLX, AAPL, TSLA)")
    parser.add_argument("--output", type=str, default="outputs/narrative_eval.json", help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(ticker=args.ticker, output_path=args.output)
