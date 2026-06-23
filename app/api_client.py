"""
Pure HTTP client — the only file in app/ allowed to talk to the backend.
Streamlit pages import from here; never import src.* directly.

Set API_BASE_URL in .env or Streamlit secrets:
  API_BASE_URL=http://localhost:8000
"""
from __future__ import annotations
import os
import time
import httpx
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
def _base() -> str:
    # Streamlit Cloud: set in app secrets as [general] API_BASE_URL
    try:
        return st.secrets["general"]["API_BASE_URL"]
    except Exception:
        return os.getenv("API_BASE_URL", "http://localhost:8000")


def _headers() -> dict:
    try:
        key = st.secrets["general"].get("API_KEY", "")
    except Exception:
        key = os.getenv("API_KEY", "")
    return {"X-API-Key": key} if key else {}


# ── Sync HTTP helpers (httpx sync — works in Streamlit) ───────────────────────
def _get(path: str, params: dict | None = None, timeout: float = 10.0) -> dict:
    url = f"{_base()}{path}"
    try:
        r = httpx.get(url, params=params, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict, timeout: float = 15.0) -> dict:
    url = f"{_base()}{path}"
    try:
        r = httpx.post(url, json=body, headers=_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "detail": e.response.text}
    except Exception as e:
        return {"error": str(e)}


# ── Health ────────────────────────────────────────────────────────────────────
def health() -> dict:
    return _get("/health")


# ── Market data ───────────────────────────────────────────────────────────────
def get_ohlcv(ticker: str, period: str = "2y") -> dict:
    return _get("/market/ohlcv", params={"ticker": ticker, "period": period})


def get_indicators(ticker: str, period: str = "1y") -> dict:
    return _get("/market/indicators", params={"ticker": ticker, "period": period})


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(rows: list[dict], ticker: str = "NFLX") -> dict:
    return _post("/predict", {"rows": rows, "ticker": ticker})


def get_live_input(ticker: str) -> dict:
    return _get("/market/live_input", params={"ticker": ticker, "n": 10})


# ── Risk ──────────────────────────────────────────────────────────────────────
def compute_risk_position(payload: dict) -> dict:
    return _post("/risk/position", payload)


def compute_risk_matrix(payload: dict) -> dict:
    return _post("/risk/matrix", payload)


# ── Execution ─────────────────────────────────────────────────────────────────
def execute_trade(payload: dict) -> dict:
    return _post("/execute", payload)


# ── Async tasks (Celery-backed heavy jobs) ────────────────────────────────────
def submit_backtest(ticker: str, days: int = 90) -> dict:
    """Returns {job_id: ...} immediately."""
    return _post("/api/v1/tasks/backtest",
                 {"ticker": ticker, "days": days})


def submit_paper_trade(ticker: str, days: int = 90) -> dict:
    return _post("/api/v1/tasks/paper_trade",
                 {"ticker": ticker, "days": days})


def submit_drift_check(ticker: str) -> dict:
    return _post("/api/v1/tasks/drift",
                 {"ticker": ticker})


def get_task_result(job_id: str) -> dict:
    """Poll task status. Returns {state, result} when done."""
    return _get(f"/api/v1/tasks/{job_id}")


def poll_until_done(job_id: str, placeholder,
                    timeout_s: int = 120, interval_s: float = 2.0) -> dict:
    """
    Block (with spinner updates) until job finishes or timeout.
    placeholder: a st.empty() element for status updates.
    """
    elapsed = 0
    while elapsed < timeout_s:
        result = get_task_result(job_id)
        state  = result.get("state", "PENDING")
        if state == "SUCCESS":
            placeholder.success("Done!")
            return result.get("result", {})
        if state == "FAILURE":
            placeholder.error(f"Task failed: {result.get('error', 'unknown')}")
            return {}
        placeholder.info(f"Running… ({elapsed}s) — state: {state}")
        time.sleep(interval_s)
        elapsed += interval_s
    placeholder.warning("Task timed out. Try again.")
    return {}


# ── Metrics / model info ──────────────────────────────────────────────────────
def get_model_info() -> dict:
    return _get("/model_info")


def get_sentiment(ticker: str = "NFLX") -> dict:
    return _get("/sentiment", params={"ticker": ticker})


def get_drift_report(ticker: str) -> dict:
    return _get("/drift", params={"ticker": ticker})


def get_feature_importance() -> dict:
    return _get("/explainability/importance")
