"""
News sentiment scoring for Netflix using Yahoo Finance RSS Feed + VADER.
Adds a rolling sentiment score as a feature.
No API key required — uses VADER (rule-based, offline) and standard RSS XML parsing.
"""
from __future__ import annotations
import logging
import xml.etree.ElementTree as ET
import pandas as pd
import requests

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
    Fetch recent news via Yahoo Finance RSS feed and score with VADER.
    Returns a daily sentiment series (mean compound score per day).
    Falls back to zeros if unavailable.
    """
    ticker = ticker.upper()
    url = f"https://feeds.finance.yahoo.com/rss.xml?s={ticker}"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            logger.warning(f"Yahoo RSS feed returned status code {resp.status_code}")
            return pd.Series(dtype=float)
            
        root = ET.fromstring(resp.content)
        records = []
        
        for item in root.findall(".//item"):
            title_node = item.find("title")
            pub_date_node = item.find("pubDate")
            
            if title_node is None or pub_date_node is None:
                continue
                
            title = title_node.text or ""
            pub_date_str = pub_date_node.text or ""
            
            try:
                # Parse date like "Tue, 23 Jun 2026 12:00:00 GMT" or "23 Jun 2026 12:00:00 -0400"
                # pandas to_datetime handles a wide variety of formats natively
                ts = pd.to_datetime(pub_date_str)
            except Exception:
                ts = pd.Timestamp.now()
                
            score = _vader_score(title)
            records.append({"date": ts.tz_localize(None).normalize(), "score": score})
            
        if not records:
            return pd.Series(dtype=float)
            
        df = pd.DataFrame(records)
        daily = df.groupby("date")["score"].mean()
        logger.info(f"Fetched {len(records)} RSS news items, {len(daily)} unique days for {ticker}")
        return daily

    except Exception as e:
        logger.warning(f"RSS Sentiment fetch failed: {e}")
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
