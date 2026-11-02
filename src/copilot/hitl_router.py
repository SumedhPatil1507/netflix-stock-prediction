"""
HITLRouter — Human-In-The-Loop gate for the Research Copilot.

Every BUY signal whose position value exceeds *threshold_usd* is held
as "pending" until a human approves or rejects it.  Persistence uses
two files:
  - outputs/hitl_pending.json   (mutable, current state)
  - logs/hitl_approvals.jsonl   (append-only audit trail)
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class HITLRouter:
    """
    Routes signals through a human approval gate.

    Parameters
    ----------
    threshold_usd  : Position value (USD) above which a BUY requires HITL.
    pending_path   : Path to the mutable pending-signals JSON file.
    approvals_log  : Path to the append-only approvals JSONL audit file.
    """

    def __init__(
        self,
        threshold_usd: float = 10_000.0,
        pending_path: str = "outputs/hitl_pending.json",
        approvals_log: str = "logs/hitl_approvals.jsonl",
    ) -> None:
        self.threshold_usd = threshold_usd
        self.pending_path = pending_path
        self.approvals_log = approvals_log

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(pending_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(approvals_log)), exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(
        self,
        signal_id: str,
        ticker: str,
        signal: str,
        position_value: float,
        strategy_name: str,
    ) -> dict[str, Any]:
        """
        Decide whether a signal requires human approval.

        Parameters
        ----------
        signal_id      : Unique identifier for this signal event.
        ticker         : Stock symbol.
        signal         : "BUY" | "HOLD" | "SELL".
        position_value : Notional USD value of the proposed position.
        strategy_name  : Name of the originating strategy.

        Returns
        -------
        dict with keys:
            ``status``       — "pending" if HITL required, "approved" otherwise.
            ``signal_id``    — echoed back (or generated if not supplied).
            ``requires_hitl``— bool.
        """
        try:
            _requires = signal == "BUY" and position_value > self.threshold_usd
            _status = "pending" if _requires else "approved"

            record: dict[str, Any] = {
                "signal_id": signal_id,
                "ticker": ticker,
                "signal": signal,
                "position_value": position_value,
                "strategy_name": strategy_name,
                "status": _status,
                "requires_hitl": _requires,
                "created_at": datetime.now().isoformat(),
            }

            if _requires:
                pending = self._load_pending()
                pending[signal_id] = record
                self._save_pending(pending)
                logger.info(
                    "HITL gate: signal %s for %s ($%.0f) is PENDING human approval.",
                    signal_id,
                    ticker,
                    position_value,
                )
            else:
                logger.info(
                    "HITL gate: signal %s for %s auto-approved (below threshold or not BUY).",
                    signal_id,
                    ticker,
                )

            return {
                "status": _status,
                "signal_id": signal_id,
                "requires_hitl": _requires,
            }

        except Exception as exc:
            logger.error("HITLRouter.route failed: %s", exc)
            return {"status": "error", "signal_id": signal_id, "requires_hitl": False}

    def approve(self, signal_id: str, approver: str = "human") -> dict[str, Any]:
        """
        Mark *signal_id* as approved.

        Updates the pending JSON and appends an audit record to the JSONL log.

        Returns
        -------
        The updated record dict, or ``{}`` on failure.
        """
        return self._update_status(signal_id, "approved", approver=approver)

    def reject(self, signal_id: str, reason: str = "") -> dict[str, Any]:
        """
        Mark *signal_id* as rejected.

        Parameters
        ----------
        reason : Optional human-readable rejection reason.

        Returns
        -------
        The updated record dict, or ``{}`` on failure.
        """
        return self._update_status(signal_id, "rejected", reason=reason)

    def pending_approvals(self) -> list[dict[str, Any]]:
        """
        Return all signals currently awaiting approval.

        Returns
        -------
        List of pending record dicts.  Returns ``[]`` on failure or if empty.
        """
        try:
            pending = self._load_pending()
            return [v for v in pending.values() if v.get("status") == "pending"]
        except Exception as exc:
            logger.error("pending_approvals failed: %s", exc)
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_status(
        self,
        signal_id: str,
        new_status: str,
        approver: str = "human",
        reason: str = "",
    ) -> dict[str, Any]:
        """Shared logic for approve / reject."""
        try:
            pending = self._load_pending()
            record = pending.get(signal_id)
            if record is None:
                logger.warning("signal_id %s not found in pending store.", signal_id)
                return {}

            record["status"] = new_status
            record["resolved_at"] = datetime.now().isoformat()
            record["approver"] = approver
            if reason:
                record["reason"] = reason

            pending[signal_id] = record
            self._save_pending(pending)
            self._append_audit(record)

            logger.info(
                "HITLRouter: signal %s → %s by %s.", signal_id, new_status, approver
            )
            return record

        except Exception as exc:
            logger.error("_update_status failed: %s", exc)
            return {}

    def _load_pending(self) -> dict[str, Any]:
        """Load the pending JSON; returns {} if file missing or corrupt."""
        try:
            if os.path.exists(self.pending_path):
                with open(self.pending_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            logger.warning("Could not load pending JSON (%s) — using empty dict.", exc)
        return {}

    def _save_pending(self, pending: dict[str, Any]) -> None:
        """Persist the pending dict to JSON."""
        try:
            with open(self.pending_path, "w", encoding="utf-8") as fh:
                json.dump(pending, fh, indent=2)
        except Exception as exc:
            logger.error("_save_pending failed: %s", exc)

    def _append_audit(self, record: dict[str, Any]) -> None:
        """Append one record to the JSONL audit log."""
        try:
            with open(self.approvals_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("_append_audit failed: %s", exc)


# ── Convenience factory ───────────────────────────────────────────────────────

def make_signal_id(ticker: str) -> str:
    """Generate a compact signal ID in the format ``xxxxxxxx-TICKER``."""
    return str(uuid.uuid4())[:8] + "-" + ticker
