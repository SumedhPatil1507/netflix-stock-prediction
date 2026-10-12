"""
Corpus management for earnings transcripts and financial news.
Handles document loading, processing, and preparation for vector store.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

import pandas as pd

logger = logging.getLogger(__name__)


class CorpusManager:
    """Manager for financial documents corpus."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize corpus manager.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.transcripts_dir = os.path.join(data_dir, "transcripts")
        self.news_dir = os.path.join(data_dir, "news")
        
        # Create directories if they don't exist
        os.makedirs(self.transcripts_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)
        
        logger.info(f"Corpus manager initialized with data directory: {data_dir}")
    
    def load_sample_transcripts(self, ticker: str = "NFLX") -> List[Dict[str, Any]]:
        """
        Load or create sample earnings transcripts.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of transcript documents with metadata
        """
        transcripts = []
        
        # Check if we have a transcripts file
        transcript_file = os.path.join(self.transcripts_dir, f"{ticker}_transcripts.json")
        
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r') as f:
                transcripts = json.load(f)
            logger.info(f"Loaded {len(transcripts)} transcripts from {transcript_file}")
        else:
            # Create sample transcripts based on Netflix's typical earnings call content
            sample_transcripts = [
                {
                    "text": """Netflix Q4 2024 Earnings Call:
                    CEO: "We had a strong finish to 2024 with subscriber growth exceeding expectations. Our ad-supported tier continues to gain momentum, now representing 40% of new sign-ups in key markets. Content investments in original programming are driving engagement and retention."
                    CFO: "Revenue grew 12.5% year-over-year to $8.8 billion. Operating margin expanded to 24% due to improved operational efficiency. Free cash flow was $1.9 billion, demonstrating our strong cash generation capabilities."
                    Analyst: The company's focus on original content and international expansion appears to be paying off, with Asia-Pacific region showing the strongest growth metrics.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "earnings_call",
                        "date": "2024-01-25",
                        "quarter": "Q4 2024",
                        "title": f"{ticker} Q4 2024 Earnings Call"
                    }
                },
                {
                    "text": """Netflix Q3 2024 Earnings Call:
                    CEO: "Our password-sharing crackdown has been successful, converting millions of households to paid memberships. The rollout of paid sharing in additional regions is proceeding according to plan. Our content slate for the remainder of the year is our strongest ever."
                    CFO: "Revenue increased 8% to $8.5 billion. Average revenue per member (ARM) grew 3% year-over-year. We expect continued ARPU growth as we optimize pricing in different markets and shift more members to higher-tier plans."
                    Outlook: Management raised guidance for the full year, citing stronger-than-expected subscriber additions and improving margin trends.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "earnings_call",
                        "date": "2024-10-18",
                        "quarter": "Q3 2024",
                        "title": f"{ticker} Q3 2024 Earnings Call"
                    }
                },
                {
                    "text": """Netflix Q2 2024 Earnings Call:
                    CEO: "We're seeing excellent momentum in our advertising business. The ad tier is now available in all major markets, and advertiser demand is strong. Our content strategy continues to resonate with audiences globally."
                    CFO: "Revenue reached $9.6 billion, up 16.5% from the prior year. Operating income was $2.1 billion with a 22% margin. We're investing heavily in content while maintaining healthy profitability."
                    Key highlights: Strong international growth, improving content ROI, and successful expansion of the advertising tier.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "earnings_call",
                        "date": "2024-07-18",
                        "quarter": "Q2 2024",
                        "title": f"{ticker} Q2 2024 Earnings Call"
                    }
                },
                {
                    "text": """Netflix Q1 2024 Earnings Call:
                    CEO: "We started the year with strong momentum, adding 9.3 million paid memberships. Our content strategy is working, with shows like '3 Body Problem' and 'Griselda' driving significant engagement. The ad-supported tier continues to grow rapidly."
                    CFO: "Revenue grew 15% to $9.4 billion. Operating margin was 26%, up from 21% in the prior year. We're confident in our ability to continue growing revenue while investing in content."
                    Market reaction: The stock responded positively to the strong subscriber growth and improving profitability metrics.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "earnings_call",
                        "date": "2024-04-23",
                        "quarter": "Q1 2024",
                        "title": f"{ticker} Q1 2024 Earnings Call"
                    }
                }
            ]
            
            # Save sample transcripts
            with open(transcript_file, 'w') as f:
                json.dump(sample_transcripts, f, indent=2)
            
            transcripts = sample_transcripts
            logger.info(f"Created {len(transcripts)} sample transcripts")
        
        return transcripts
    
    def load_sample_news(self, ticker: str = "NFLX") -> List[Dict[str, Any]]:
        """
        Load or create sample financial news articles.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news articles with metadata
        """
        news = []
        
        # Check if we have a news file
        news_file = os.path.join(self.news_dir, f"{ticker}_news.json")
        
        if os.path.exists(news_file):
            with open(news_file, 'r') as f:
                news = json.load(f)
            logger.info(f"Loaded {len(news)} news articles from {news_file}")
        else:
            # Create sample news articles
            sample_news = [
                {
                    "text": """Netflix Stock Surges on Strong Subscriber Growth
                    Netflix shares jumped 5% in after-hours trading after the company reported better-than-expected subscriber growth. The streaming giant added 9.3 million paid memberships in Q1 2024, significantly above analyst estimates of 6 million. The company's ad-supported tier continues to gain traction, now accounting for 30% of new sign-ups in key markets. Analysts remain bullish on the stock, citing improving fundamentals and successful content strategy.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "news",
                        "date": "2024-04-24",
                        "publication": "Financial Times",
                        "title": f"{ticker} Stock Surges on Strong Subscriber Growth"
                    }
                },
                {
                    "text": """Netflix Faces Increasing Competition in Streaming Wars
                    The streaming landscape is becoming increasingly competitive with Disney+, Amazon Prime Video, and Apple TV+ investing heavily in original content. However, Netflix maintains its leadership position with over 270 million global subscribers. The company's focus on diverse content and international expansion provides a competitive moat. Some analysts express concern about rising content costs and market saturation in developed regions.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "news",
                        "date": "2024-06-15",
                        "publication": "Bloomberg",
                        "title": f"{ticker} Faces Increasing Competition in Streaming Wars"
                    }
                },
                {
                    "text": """Netflix's Advertising Business Shows Promise
                    Netflix's advertising-supported tier is showing strong growth metrics, with average revenue per user (ARPU) trending higher than expected. The company has partnered with major advertisers and is improving its ad targeting capabilities. This new revenue stream could provide a significant boost to margins as it scales. However, some investors worry that ads might negatively impact user experience and lead to higher churn rates.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "news",
                        "date": "2024-08-20",
                        "publication": "Reuters",
                        "title": f"{ticker}'s Advertising Business Shows Promise"
                    }
                },
                {
                    "text": """Wall Street Remains Bullish on Netflix Despite Valuation Concerns
                    Despite trading at high multiples, Netflix continues to attract bullish sentiment from Wall Street analysts. The company's dominant market position, strong cash flow generation, and international growth opportunities justify the premium valuation according to bulls. However, some analysts have raised concerns about slowing growth in North America and increasing content costs. The stock's performance will depend on execution of the advertising strategy and continued content success.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "news",
                        "date": "2024-09-10",
                        "publication": "CNBC",
                        "title": "Wall Street Remains Bullish on Netflix Despite Valuation Concerns"
                    }
                },
                {
                    "text": """Netflix's Content Strategy Drives International Expansion
                    Netflix's investment in local-language content is paying off with strong growth in Asia-Pacific and Latin American markets. The company's strategy of producing region-specific content while maintaining global appeal has proven successful. International markets now account for over 60% of Netflix's subscriber base. This geographic diversification reduces dependence on any single market and provides long-term growth opportunities as internet penetration increases globally.""",
                    "metadata": {
                        "ticker": ticker,
                        "source": "news",
                        "date": "2024-07-05",
                        "publication": "Wall Street Journal",
                        "title": f"{ticker}'s Content Strategy Drives International Expansion"
                    }
                }
            ]
            
            # Save sample news
            with open(news_file, 'w') as f:
                json.dump(sample_news, f, indent=2)
            
            news = sample_news
            logger.info(f"Created {len(news)} sample news articles")
        
        return news
    
    def prepare_documents_for_vector_store(
        self,
        ticker: str = "NFLX"
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Prepare all documents for insertion into vector store.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (documents, metadatas, ids)
        """
        transcripts = self.load_sample_transcripts(ticker)
        news = self.load_sample_news(ticker)
        
        all_docs = transcripts + news
        
        documents = [doc["text"] for doc in all_docs]
        metadatas = [doc["metadata"] for doc in all_docs]
        ids = [
            f"{doc['metadata']['source']}_{doc['metadata']['date']}_{i}"
            for i, doc in enumerate(all_docs)
        ]
        
        logger.info(f"Prepared {len(documents)} documents for vector store")
        return documents, metadatas, ids
    
    def add_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_type: str = "news"
    ) -> None:
        """
        Add a single document to the corpus.
        
        Args:
            text: Document text
            metadata: Document metadata
            doc_type: Type of document (news or transcript)
        """
        if doc_type == "news":
            file_path = os.path.join(self.news_dir, f"{metadata.get('ticker', 'NFLX')}_news.json")
        else:
            file_path = os.path.join(self.transcripts_dir, f"{metadata.get('ticker', 'NFLX')}_transcripts.json")
        
        # Load existing documents
        existing_docs = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_docs = json.load(f)
        
        # Add new document
        existing_docs.append({"text": text, "metadata": metadata})
        
        # Save updated documents
        with open(file_path, 'w') as f:
            json.dump(existing_docs, f, indent=2)
        
        logger.info(f"Added document to {doc_type} corpus")