"""
CopilotWriter — synthesises a cited research note from model state,
risk parameters, and retrieved document context.

Uses Groq llama3-8b-8192 when GROQ_API_KEY is set and the ``groq``
package is installed.  Falls back to a deterministic rule-based
template otherwise — the fallback always returns a valid string.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional Groq import ──────────────────────────────────────────────────────
try:
    import groq as _groq_module  # noqa: F401
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.info("groq package not installed — CopilotWriter will use rule-based fallback.")


class CopilotWriter:
    """
    Writes a one-page research note given model state, risk state, and
    retrieved document excerpts.

    Parameters
    ----------
    model : Groq model name (only used when GROQ_AVAILABLE and key is set).
    """

    def __init__(self, model: str = "llama3-8b-8192") -> None:
        self.model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def write_note(
        self,
        tool_state: dict[str, Any],
        retrieval: dict[str, Any],
        ticker: str,
    ) -> str:
        """
        Synthesise a research note combining model signal, risk state,
        and retrieved document context.

        Parameters
        ----------
        tool_state : Output of :class:`CopilotToolAgent` (pred_return, signal, …).
        retrieval  : Output of :class:`CopilotRetriever.retrieve` (documents list).
        ticker     : Stock symbol.

        Returns
        -------
        A non-empty string research note.  Never raises.
        """
        try:
            if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
                return self._groq_note(tool_state, retrieval, ticker)
        except Exception as exc:
            logger.warning("Groq note generation failed (%s) — using fallback.", exc)

        return self._rule_based_note(tool_state, retrieval, ticker)

    # ── Groq path ─────────────────────────────────────────────────────────────

    def _groq_note(
        self,
        tool_state: dict[str, Any],
        retrieval: dict[str, Any],
        ticker: str,
    ) -> str:
        """Call Groq llama3-8b-8192 to generate the research note."""
        import groq

        pred_return = tool_state.get("pred_return", 0.0)
        signal = tool_state.get("signal", "HOLD")
        lo = tool_state.get("conformal_lo", 0.0)
        hi = tool_state.get("conformal_hi", 0.0)
        stop_loss = tool_state.get("stop_loss", "N/A")
        kelly = tool_state.get("kelly_fraction", 0.0)

        shap_drivers = tool_state.get("shap_drivers", [])
        top_driver = (
            shap_drivers[0]["feature"] if shap_drivers else "unknown feature"
        )

        docs = retrieval.get("documents", [])[:3]
        context_block = _format_context(docs)

        prompt = (
            f"Write a concise one-page equity research note for {ticker}.\n\n"
            f"Model signal: {signal}\n"
            f"Predicted next-day return: {pred_return:+.3f}%\n"
            f"Conformal interval: [{lo:.3f}%, {hi:.3f}%]\n"
            f"Primary feature driver: {top_driver}\n"
            f"Risk: stop_loss={stop_loss}, kelly_fraction={kelly:.3f}\n\n"
            f"Supporting context from recent filings and news:\n{context_block}\n\n"
            "Structure: Executive Summary, Key Drivers (with citations), "
            "Risk Factors, Conclusion. Be concise and cite sources inline."
        )

        client = groq.Groq()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        note = response.choices[0].message.content or ""
        logger.info("Groq note generated for %s (%d chars).", ticker, len(note))
        return note

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _rule_based_note(
        self,
        tool_state: dict[str, Any],
        retrieval: dict[str, Any],
        ticker: str,
    ) -> str:
        """Deterministic structured-string research note (no external API)."""
        pred_return = tool_state.get("pred_return", 0.0)
        signal = tool_state.get("signal", "HOLD")
        lo = tool_state.get("conformal_lo", 0.0)
        hi = tool_state.get("conformal_hi", 0.0)
        stop_loss = tool_state.get("stop_loss", "N/A")
        kelly = tool_state.get("kelly_fraction", 0.0)
        is_halted = tool_state.get("is_halted", False)

        shap_drivers: list[dict] = tool_state.get("shap_drivers", [])
        top_driver = shap_drivers[0]["feature"] if shap_drivers else "RSI"
        driver_lines = "\n".join(
            f"  - {d['feature']}: importance={d['importance']:.4f}"
            for d in shap_drivers[:3]
        ) or "  - No feature importance data available."

        docs = retrieval.get("documents", [])[:3]
        excerpt_lines = ""
        for doc in docs:
            meta = doc.get("metadata", {})
            title = meta.get("title", "Unnamed Source")
            date = meta.get("date", "Unknown Date")
            snippet = doc.get("text", "")[:160].replace("\n", " ")
            excerpt_lines += f"  - {snippet}... [cited from: {title}, {date}]\n"
        if not excerpt_lines:
            excerpt_lines = "  - No contextual documents retrieved.\n"

        halt_note = " ⚠️ TRADING HALTED (circuit breaker active)." if is_halted else ""

        note = (
            f"=== Alpha Engine Pro — Research Note: {ticker} ===\n\n"
            f"Executive Summary:\n"
            f"  Model is {signal.lower()} on {ticker}: predicted return "
            f"{pred_return:+.3f}%, conformal interval "
            f"[{lo:.3f}%, {hi:.3f}%].{halt_note}\n\n"
            f"Key Drivers:\n"
            f"{driver_lines}\n\n"
            f"Risk Parameters:\n"
            f"  stop_loss={stop_loss}, kelly_fraction={kelly:.3f}\n\n"
            f"Supporting Context:\n"
            f"{excerpt_lines}\n"
            f"Conclusion:\n"
            f"  Primary driver is {top_driver}. Monitor conformal interval width "
            f"for confidence signal. Human review required for positions above "
            f"threshold."
        )
        return note


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[dict[str, Any]]) -> str:
    """Format retrieved docs into a compact string for the LLM prompt."""
    if not docs:
        return "(no context documents available)"
    lines = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        title = meta.get("title", "Source")
        date = meta.get("date", "")
        snippet = doc.get("text", "")[:300].replace("\n", " ")
        lines.append(f"[{i}] {title} ({date}): {snippet}...")
    return "\n".join(lines)
