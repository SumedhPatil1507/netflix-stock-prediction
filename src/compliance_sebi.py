"""
SEBI Algorithmic Trading Compliance Module — Alpha Engine Pro.

Maps to SEBI's algorithmic trading circular requirements:
  - Audit trail (complete order log)
  - Kill switch / circuit-breaker documentation
  - Order-to-trade ratio monitoring
  - Pre-trade risk checks
  - Position limits
  - Human-in-the-loop gate
  - Model version traceability
  - Latency logging

Each check returns a dict with:
    check_id, description, status (PASS | FAIL | WARN),
    evidence (str), remediation (str)

This structure mirrors the evidence+remediation JSON-report pattern.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
_AUDIT_LOG_PATH = "logs/audit_trail.jsonl"
_KILL_SWITCH_PATH = "outputs/kill_switch.json"


class SEBIComplianceChecker:
    """
    Runs 8 SEBI algorithmic-trading compliance checks and generates a
    structured JSON report.

    Parameters
    ----------
    audit_log_path    : Path to the append-only order audit trail JSONL.
    kill_switch_path  : Path to the kill-switch state JSON.
    """

    def __init__(
        self,
        audit_log_path: str = _AUDIT_LOG_PATH,
        kill_switch_path: str = _KILL_SWITCH_PATH,
    ) -> None:
        self.audit_log_path = audit_log_path
        self.kill_switch_path = kill_switch_path

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(audit_log_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(kill_switch_path)), exist_ok=True)

    # ── Order logging ──────────────────────────────────────────────────────────

    def log_order(self, order_dict: dict[str, Any]) -> None:
        """
        Append one order record to the audit trail JSONL.

        Parameters
        ----------
        order_dict : Any dict representing an order.  A ``timestamp`` and
                     ``order_id`` are injected if missing.
        """
        try:
            record = dict(order_dict)
            if "timestamp" not in record:
                record["timestamp"] = datetime.now().isoformat()
            if "order_id" not in record:
                import uuid
                record["order_id"] = str(uuid.uuid4())[:8]

            with open(self.audit_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

            logger.info("Audit trail: order %s logged.", record.get("order_id"))
        except Exception as exc:
            logger.error("log_order failed: %s", exc)

    # ── Kill switch ────────────────────────────────────────────────────────────

    def get_kill_switch_status(self) -> dict[str, Any]:
        """
        Read current kill-switch state.

        Returns
        -------
        dict with keys: ``active`` (bool), ``reason`` (str|None),
        ``activated_at`` (str|None).
        """
        default: dict[str, Any] = {
            "active": False,
            "reason": None,
            "activated_at": None,
        }
        try:
            if os.path.exists(self.kill_switch_path):
                with open(self.kill_switch_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                default.update(data)
        except Exception as exc:
            logger.warning("get_kill_switch_status failed (%s) — returning default.", exc)
        return default

    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate the kill switch with *reason*.  Writes kill_switch.json.
        """
        try:
            state = {
                "active": True,
                "reason": reason,
                "activated_at": datetime.now().isoformat(),
            }
            with open(self.kill_switch_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            logger.warning("Kill switch ACTIVATED. Reason: %s", reason)
        except Exception as exc:
            logger.error("activate_kill_switch failed: %s", exc)

    def deactivate_kill_switch(self) -> None:
        """
        Deactivate the kill switch.  Sets ``active`` to False.
        """
        try:
            state = self.get_kill_switch_status()
            state["active"] = False
            state["deactivated_at"] = datetime.now().isoformat()
            with open(self.kill_switch_path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
            logger.info("Kill switch deactivated.")
        except Exception as exc:
            logger.error("deactivate_kill_switch failed: %s", exc)

    # ── Order-to-trade ratio ───────────────────────────────────────────────────

    def compute_order_to_trade_ratio(self, window_days: int = 30) -> float:
        """
        Calculate order-to-trade ratio over *window_days*.

        SEBI guideline: ratio <= 50 (orders per executed trade).

        Returns
        -------
        float ratio (orders / max(1, buy_trades)).  Returns 0.0 if no data.
        """
        try:
            if not os.path.exists(self.audit_log_path):
                return 0.0

            cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
            orders = 0
            trades = 0

            with open(self.audit_log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("timestamp", "")
                    if ts >= cutoff:
                        orders += 1
                        if rec.get("signal", "").upper() == "BUY":
                            trades += 1

            return float(orders) / max(1, trades)

        except Exception as exc:
            logger.error("compute_order_to_trade_ratio failed: %s", exc)
            return 0.0

    # ── Report ─────────────────────────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """
        Run all 8 SEBI compliance checks and return a structured report.

        Returns
        -------
        dict with keys:
            ``generated_at`` (ISO str), ``overall_status`` ("PASS" | "FAIL"),
            ``summary`` (str), ``checks`` (list[dict]).
        """
        checks: list[dict[str, Any]] = [
            self._check_audit_trail(),
            self._check_kill_switch(),
            self._check_order_trade_ratio(),
            self._check_pre_trade_risk(),
            self._check_position_limit(),
            self._check_hitl_gate(),
            self._check_model_traceability(),
            self._check_latency_logging(),
        ]

        statuses = [c["status"] for c in checks]
        overall = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"

        fail_ids = [c["check_id"] for c in checks if c["status"] != "PASS"]
        summary = (
            f"{sum(1 for s in statuses if s == 'PASS')}/{len(statuses)} checks passed."
            + (f" Failed: {', '.join(fail_ids)}." if fail_ids else " All checks passed.")
        )

        return {
            "generated_at": datetime.now().isoformat(),
            "overall_status": overall,
            "summary": summary,
            "checks": checks,
        }

    # ── Individual checks ──────────────────────────────────────────────────────

    def _check_audit_trail(self) -> dict[str, Any]:
        """SEBI_001 — Audit trail exists and has entries."""
        check_id = "SEBI_001"
        description = "Audit trail JSONL exists and contains order records."
        try:
            if not os.path.exists(self.audit_log_path):
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"{self.audit_log_path} does not exist.",
                    remediation=(
                        "Call SEBIComplianceChecker.log_order() for every order "
                        "to create the audit trail at " + self.audit_log_path + "."
                    ),
                )
            count = _count_jsonl_lines(self.audit_log_path)
            if count == 0:
                return _check_result(
                    check_id, description, "WARN",
                    evidence=f"{self.audit_log_path} exists but is empty.",
                    remediation="Ensure all orders are logged via log_order().",
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"{self.audit_log_path} exists with {count} records.",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify audit_log_path is writable.",
            )

    def _check_kill_switch(self) -> dict[str, Any]:
        """SEBI_002 — Kill switch JSON exists with required fields."""
        check_id = "SEBI_002"
        description = "Kill-switch / circuit-breaker config file is present with required fields."
        required_fields = {"active", "reason", "activated_at"}
        try:
            if not os.path.exists(self.kill_switch_path):
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"{self.kill_switch_path} does not exist.",
                    remediation=(
                        "Call activate_kill_switch() or deactivate_kill_switch() "
                        "to create the file, or deploy with a default kill_switch.json."
                    ),
                )
            with open(self.kill_switch_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            missing = required_fields - set(data.keys())
            if missing:
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"Missing fields: {missing}.",
                    remediation="Regenerate kill_switch.json via activate/deactivate methods.",
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"Kill-switch state: active={data.get('active')}.",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify kill_switch_path and JSON format.",
            )

    def _check_order_trade_ratio(self) -> dict[str, Any]:
        """SEBI_003 — Order-to-trade ratio <= 50 per SEBI guidelines."""
        check_id = "SEBI_003"
        description = "Order-to-trade ratio within SEBI limit of 50."
        try:
            ratio = self.compute_order_to_trade_ratio()
            if ratio > 50:
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"Ratio = {ratio:.1f} (limit: 50).",
                    remediation=(
                        "Review algo logic for excessive order generation. "
                        "Implement rate limiting and reduce cancellation frequency."
                    ),
                )
            if ratio == 0.0:
                return _check_result(
                    check_id, description, "WARN",
                    evidence="No orders in audit trail — ratio is 0 (no data).",
                    remediation="Ensure orders are being logged via log_order().",
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"Ratio = {ratio:.2f} (limit: 50).",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify audit trail is accessible.",
            )

    def _check_pre_trade_risk(self) -> dict[str, Any]:
        """SEBI_004 — Pre-trade risk check module is available."""
        check_id = "SEBI_004"
        description = "Pre-trade risk check module (RiskManager) is importable and functional."
        try:
            from src.risk_manager import RiskManager  # noqa: F401
            return _check_result(
                check_id, description, "PASS",
                evidence="src.risk_manager.RiskManager imported successfully.",
                remediation="No action required.",
            )
        except ImportError as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation=(
                    "Ensure src/risk_manager.py is present and all dependencies installed."
                ),
            )

    def _check_position_limit(self) -> dict[str, Any]:
        """SEBI_005 — Max position per instrument <= 10% of portfolio (SEBI algo norm)."""
        check_id = "SEBI_005"
        description = "Max position per instrument <= 10% of portfolio (SEBI algo norm)."
        try:
            from src.risk_manager import RiskConfig
            cfg = RiskConfig()
            limit = cfg.max_position_pct
            if limit > 0.10:
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"max_position_pct={limit*100:.0f}% exceeds 10% SEBI limit.",
                    remediation=(
                        "Set RiskConfig.max_position_pct <= 0.10 and redeploy."
                    ),
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"max_position_pct={limit*100:.0f}% (limit: 10%).",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify src/risk_manager.py RiskConfig default values.",
            )

    def _check_hitl_gate(self) -> dict[str, Any]:
        """SEBI_006 — HITL gate is in place for large orders."""
        check_id = "SEBI_006"
        description = "HITL gate (HITLRouter) is deployed and pending-signals file is accessible."
        hitl_pending = "outputs/hitl_pending.json"
        try:
            from src.copilot.hitl_router import HITLRouter  # noqa: F401
            # The file may not exist yet (no signals routed yet) — that is WARN not FAIL
            if not os.path.exists(hitl_pending):
                return _check_result(
                    check_id, description, "WARN",
                    evidence=(
                        f"HITLRouter importable but {hitl_pending} not yet created. "
                        "No signals have been routed through the HITL gate yet."
                    ),
                    remediation=(
                        "Route at least one signal through HITLRouter.route() to "
                        "initialise the pending-signals file."
                    ),
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"HITLRouter importable; {hitl_pending} exists.",
                remediation="No action required.",
            )
        except ImportError as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation=(
                    "Ensure src/copilot/hitl_router.py is present and importable."
                ),
            )

    def _check_model_traceability(self) -> dict[str, Any]:
        """SEBI_007 — Model version registry exists with at least one entry."""
        check_id = "SEBI_007"
        description = "Model version registry (models/registry.json) has at least one entry."
        registry_path = "models/registry.json"
        try:
            if not os.path.exists(registry_path):
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"{registry_path} does not exist.",
                    remediation=(
                        "Run python main.py to train a model and register it, or "
                        "call src.model_registry.save_versioned_model()."
                    ),
                )
            with open(registry_path, "r", encoding="utf-8") as fh:
                reg = json.load(fh)
            n = len(reg.get("models", []))
            if n == 0:
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"{registry_path} exists but has 0 model versions.",
                    remediation="Train and register at least one model version.",
                )
            return _check_result(
                check_id, description, "PASS",
                evidence=f"{registry_path} has {n} registered model version(s).",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify models/registry.json format.",
            )

    def _check_latency_logging(self) -> dict[str, Any]:
        """SEBI_008 — Agent latency / trace log exists."""
        check_id = "SEBI_008"
        description = "Agent trace / latency log (logs/agent_traces.jsonl) exists."
        traces_path = "logs/agent_traces.jsonl"
        try:
            if not os.path.exists(traces_path):
                return _check_result(
                    check_id, description, "FAIL",
                    evidence=f"{traces_path} does not exist.",
                    remediation=(
                        "Ensure src/agent_traces.py is used for logging all "
                        "agent invocations so latency is captured."
                    ),
                )
            count = _count_jsonl_lines(traces_path)
            return _check_result(
                check_id, description, "PASS",
                evidence=f"{traces_path} exists with {count} entries.",
                remediation="No action required.",
            )
        except Exception as exc:
            return _check_result(
                check_id, description, "FAIL",
                evidence=str(exc),
                remediation="Verify logs/agent_traces.jsonl is writable.",
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_result(
    check_id: str,
    description: str,
    status: str,
    evidence: str,
    remediation: str,
) -> dict[str, Any]:
    """Build a standardised check result dict."""
    return {
        "check_id": check_id,
        "description": description,
        "status": status,
        "evidence": evidence,
        "remediation": remediation,
    }


def _count_jsonl_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count
