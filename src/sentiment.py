"""
News sentiment scoring for Netflix using yfinance news + VADER.
Adds a rolling sentiment score as a feature.
No API key required — uses VADER (rule-based, offline).
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _vader_score(text: str) -> float:
    """Return compound VADER sentiment score [-1, 1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except ImportError:
        return 0.0


def fetch_sentiment(ticker: str = "NFLX", days: int = 90) -> pd.Series:
    """
    Fetch recent news via yfinance and score with VADER.
    Returns a daily sentiment series (mean compound score per day).
    Falls back to zeros if yfinance or VADER unavailable.
    """
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news
        if not news:
            return pd.Series(dtype=float)

        records = []
        for item in news:
            ts    = pd.Timestamp(item.get("providerPublishTime", 0), unit="s")
            title = item.get("title", "")
            score = _vader_score(title)
            records.append({"date": ts.normalize(), "score": score})

        df = pd.DataFrame(records)
        daily = df.groupby("date")["score"].mean()
        logger.info(f"Fetched {len(records)} news items, {len(daily)} unique days")
        return daily

    except Exception as e:
        logger.warning(f"Sentiment fetch failed: {e}")
        return pd.Series(dtype=float)


def add_sentiment_features(df: pd.DataFrame,
                            ticker: str = "NFLX") -> pd.DataFrame:
    """
    Adds Sentiment_1d and Sentiment_3d (rolling mean) columns.
    Safe to call even if sentiment fetch fails — fills with 0.
    """
    df = df.copy()
    sentiment = fetch_sentiment(ticker)

    if sentiment.empty:
        df["Sentiment_1d"] = 0.0
        df["Sentiment_3d"] = 0.0
        return df

    # Align to df index
    sent_aligned = sentiment.reindex(df.index, method="ffill").fillna(0)
    df["Sentiment_1d"] = sent_aligned
    df["Sentiment_3d"] = sent_aligned.rolling(3, min_periods=1).mean()
    logger.info("Sentiment features added")
    return df
