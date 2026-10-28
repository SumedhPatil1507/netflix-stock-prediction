"""
Retriever agent for RAG system.
Performs retrieval over ChromaDB vector store to find relevant financial documents.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .vector_store import VectorStore
from .corpus import CorpusManager

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Agent that retrieves relevant financial documents using RAG."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        corpus_manager: CorpusManager
    ):
        """
        Initialize retriever agent.
        
        Args:
            vector_store: ChromaDB vector store instance
            corpus_manager: Corpus manager instance
        """
        self.vector_store = vector_store
        self.corpus_manager = corpus_manager
        
        logger.info("Retriever agent initialized")
    
    def retrieve_documents(
        self,
        query: str,
        ticker: str = "NFLX",
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: Query text
            ticker: Stock ticker symbol
            n_results: Number of results to return
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        # Query vector store with ticker filter
        results = self.vector_store.query(
            query_text=query,
            n_results=n_results,
            where={"ticker": ticker}
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            retrieved_docs.append(doc)
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        
        return {
            "query": query,
            "ticker": ticker,
            "documents": retrieved_docs,
            "timestamp": datetime.now().isoformat()
        }
    
    def retrieve_by_date_range(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve documents within a specific date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            n_results: Number of results to return
            
        Returns:
            Dictionary containing retrieved documents
        """
        # Get all documents for ticker
        all_results = self.vector_store.get_by_ticker(ticker, n_results=n_results * 2)
        
        # Filter by date range
        filtered_docs = []
        for i in range(len(all_results['ids'])):
            metadata = all_results['metadatas'][i]
            doc_date = metadata.get('date', '')
            
            if start_date <= doc_date <= end_date:
                filtered_docs.append({
                    "id": all_results['ids'][i],
                    "text": all_results['documents'][i],
                    "metadata": metadata
                })
        
        # Limit results
        filtered_docs = filtered_docs[:n_results]
        
        logger.info(f"Retrieved {len(filtered_docs)} documents for {ticker} from {start_date} to {end_date}")
        
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "documents": filtered_docs,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_documents(
        self,
        ticker: str = "NFLX",
        days: int = 30,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Get recent documents within the last N days.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            n_results: Number of results to return
            
        Returns:
            Dictionary containing recent documents
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.retrieve_by_date_range(
            ticker=ticker,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            n_results=n_results
        )
    
    def format_retrieval_results(self, results: Dict[str, Any]) -> str:
        """
        Format retrieval results for display or further processing.
        
        Args:
            results: Results from retrieve_documents
            
        Returns:
            Formatted string of retrieved documents
        """
        formatted = f"Retrieval Results for {results['ticker']}\n"
        formatted += f"Query: {results['query']}\n"
        formatted += f"Retrieved {len(results['documents'])} documents:\n\n"
        
        for i, doc in enumerate(results['documents'], 1):
            formatted += f"Document {i}:\n"
            formatted += f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            formatted += f"Date: {doc['metadata'].get('date', 'Unknown')}\n"
            formatted += f"Title: {doc['metadata'].get('title', 'N/A')}\n"
            formatted += f"Content: {doc['text'][:300]}...\n"
            if doc.get('distance'):
                formatted += f"Relevance Score: {1 - doc['distance']:.3f}\n"
            formatted += "\n"
        
        return formatted