"""
Monitoring and alerting.
Sends email or Slack notification when drift is detected
or model performance degrades.
Uses environment variables — no hardcoded credentials.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def send_slack_alert(message: str, webhook_url: Optional[str] = None) -> bool:
    """
    Send a Slack notification via webhook.
    Set SLACK_WEBHOOK_URL in .env or environment variables.
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        logger.debug("SLACK_WEBHOOK_URL not set — skipping Slack alert")
        return False
    try:
        import requests
        resp = requests.post(url, json={"text": message}, timeout=5)
        if resp.status_code == 200:
            logger.info("Slack alert sent")
            return True
        logger.warning(f"Slack alert failed: {resp.status_code}")
        return False
    except Exception as e:
        logger.warning(f"Slack alert error: {e}")
        return False


def send_email_alert(subject: str, body: str,
                     to_email: Optional[str] = None) -> bool:
    """
    Send an email alert via SMTP.
    Required env vars: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL
    """
    to    = to_email or os.getenv("ALERT_EMAIL")
    host  = os.getenv("SMTP_HOST")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER")
    pwd   = os.getenv("SMTP_PASS")

    if not all([to, host, user, pwd]):
        logger.debug("Email config incomplete — skipping email alert")
        return False

    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = user
        msg["To"]      = to

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, pwd)
            server.send_message(msg)

        logger.info(f"Email alert sent to {to}")
        return True
    except Exception as e:
        logger.warning(f"Email alert error: {e}")
        return False


def alert_drift(n_drifted: int, total: int, ticker: str = "NFLX") -> None:
    """Send drift alert if significant drift detected."""
    pct = n_drifted / total * 100 if total > 0 else 0
    msg = (f"[{ticker}] Model Drift Alert: {n_drifted}/{total} features drifted "
           f"({pct:.1f}%). Consider retraining.")
    logger.warning(msg)
    send_slack_alert(msg)
    send_email_alert(f"[{ticker}] Drift Alert", msg)


def alert_retrain_complete(ticker: str, metrics: dict, version: str) -> None:
    """Send notification when scheduled retraining completes."""
    dir_acc = metrics.get("Dir_Acc", 0)
    cv_r2   = metrics.get("CV_R2", 0)
    msg = (f"[{ticker}] Retraining complete — version: {version}\n"
           f"Dir Acc: {dir_acc:.1f}% | CV R²: {cv_r2:.4f}")
    logger.info(msg)
    send_slack_alert(msg)
