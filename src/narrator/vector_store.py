"""
ChromaDB vector store management for RAG system.
Handles document storage, retrieval, and management for earnings transcripts and financial news.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for RAG system."""
    
    def __init__(
        self,
        collection_name: str = "financial_documents",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the sentence-transformers model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embedding function
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        except Exception as e:
            logger.warning(f"Error initializing SentenceTransformerEmbeddingFunction: {e}")
            # Fallback to default embedding function
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized: {collection_name} at {persist_directory}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document text
            metadatas: List of metadata dictionaries for each document
            ids: Optional list of unique IDs for each document
        """
        if ids is None:
            ids = [f"doc_{datetime.now().timestamp()}_{i}" for i in range(len(documents))]
        
        if len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError("documents, metadatas, and ids must have the same length")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Dictionary containing query results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        logger.info(f"Query returned {len(results['ids'][0])} results")
        return results
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from vector store")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")
    
    def get_by_ticker(self, ticker: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Get documents filtered by ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            n_results: Number of results to return
            
        Returns:
            Dictionary containing filtered results
        """
        # Get all documents for the ticker
        results = self.collection.get(
            where={"ticker": ticker},
            limit=n_results
        )
        
        logger.info(f"Retrieved {len(results['ids'])} documents for ticker {ticker}")
        return results