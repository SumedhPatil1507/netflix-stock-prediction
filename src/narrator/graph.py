"""
LangGraph pipeline for AI Market Narrator.
Orchestrates retriever and synthesis agents in a multi-agent workflow.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not installed. Multi-agent workflow will be simplified.")

from .vector_store import VectorStore
from .corpus import CorpusManager
from .retriever_agent import RetrieverAgent
from .synthesis_agent import SynthesisAgent

logger = logging.getLogger(__name__)


class NarratorState(TypedDict):
    """State for the narrator workflow."""
    ticker: str
    prediction: float
    conformal_interval: tuple[float, float]
    current_price: float
    query: Optional[str]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    narrative: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    sentiment: Optional[str]
    error: Optional[str]
    timestamp: str


class NarratorGraph:
    """LangGraph workflow for AI Market Narrator."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        corpus_manager: Optional[CorpusManager] = None,
        retriever_llm: str = "gpt-4o-mini",
        synthesis_llm: str = "gpt-4o-mini"
    ):
        """
        Initialize the narrator graph.
        
        Args:
            vector_store: ChromaDB vector store instance
            corpus_manager: Corpus manager instance
            retriever_llm: Model for retriever agent
            synthesis_llm: Model for synthesis agent
        """
        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.corpus_manager = corpus_manager or CorpusManager()
        
        # Initialize agents
        self.retriever_agent = RetrieverAgent(
            vector_store=self.vector_store,
            corpus_manager=self.corpus_manager
        )
        
        self.synthesis_agent = SynthesisAgent(
            llm_model=synthesis_llm
        )
        
        # Build the graph if LangGraph is available
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_graph()
            logger.info("Narrator graph initialized successfully")
        else:
            self.graph = None
            logger.info("Narrator initialized in simplified mode (LangGraph not available)")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(NarratorState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("synthesize", END)
        workflow.add_edge("handle_error", END)
        
        # Conditional routing: if retrieval errored → handle_error, else → synthesize
        workflow.add_conditional_edges(
            "retrieve",
            self._check_retrieval_error,
            {
                "error": "handle_error",
                "continue": "synthesize"
            }
        )
        
        return workflow.compile()
    
    def _retrieve_node(self, state: NarratorState) -> NarratorState:
        """Retriever agent node."""
        try:
            ticker = state["ticker"]
            prediction = state["prediction"]
            
            # Generate query based on prediction
            sentiment = "bullish" if prediction > 0 else "bearish"
            query = f"{ticker} stock analysis {sentiment} factors earnings news"
            
            # Retrieve documents
            retrieval_results = self.retriever_agent.retrieve_documents(
                query=query,
                ticker=ticker,
                n_results=5
            )
            
            state["query"] = query
            state["retrieved_documents"] = retrieval_results["documents"]
            state["error"] = None
            
            logger.info(f"Retrieval node completed for {ticker}")
            
        except Exception as e:
            logger.error(f"Error in retrieval node: {e}")
            state["error"] = str(e)
            state["retrieved_documents"] = None
        
        return state
    
    def _synthesize_node(self, state: NarratorState) -> NarratorState:
        """Synthesis agent node."""
        try:
            ticker = state["ticker"]
            prediction = state["prediction"]
            conformal_interval = state["conformal_interval"]
            current_price = state["current_price"]
            retrieved_docs = state["retrieved_documents"] or []
            
            # Generate narrative
            narrative_result = self.synthesis_agent.generate_narrative(
                prediction=prediction,
                conformal_interval=conformal_interval,
                ticker=ticker,
                retrieved_docs=retrieved_docs,
                current_price=current_price
            )
            
            state["narrative"] = narrative_result["narrative"]
            state["citations"] = narrative_result["citations"]
            state["sentiment"] = narrative_result["sentiment"]
            state["error"] = None
            
            logger.info(f"Synthesis node completed for {ticker}")
            
        except Exception as e:
            logger.error(f"Error in synthesis node: {e}")
            state["error"] = str(e)
            state["narrative"] = None
            state["citations"] = None
            state["sentiment"] = None
        
        return state
    
    def _handle_error_node(self, state: NarratorState) -> NarratorState:
        """Error handling node."""
        error_msg = state.get("error", "Unknown error")
        logger.error(f"Handling error: {error_msg}")
        
        # Set fallback narrative
        ticker = state["ticker"]
        prediction = state["prediction"]
        conformal_interval = state["conformal_interval"]
        current_price = state["current_price"]
        
        sentiment = "bullish" if prediction > 0 else "bearish"
        strength = "strongly" if abs(prediction) > 0.02 else "moderately"
        
        fallback_narrative = f"""Executive Summary:
The model predicts a {strength} {sentiment} outlook for {ticker} with a predicted next-day return of {prediction*100:.2f}%. 
The conformal prediction interval suggests the return will likely fall between {conformal_interval[0]*100:.2f}% and {conformal_interval[1]*100:.2f}%.

Note: An error occurred during the narrative generation process: {error_msg}
This is a simplified narrative without full context from retrieved documents.

Risk Factors:
- Market volatility and unexpected news events
- Model limitations and potential overfitting
- Error in document retrieval or synthesis process

Conclusion:
The model indicates a {strength} {sentiment} sentiment for {ticker}, but please note that the full narrative generation encountered an error."""
        
        state["narrative"] = fallback_narrative
        state["citations"] = []
        state["sentiment"] = sentiment
        
        return state
    
    def _check_retrieval_error(self, state: NarratorState) -> str:
        """Check if retrieval encountered an error."""
        if state.get("error"):
            return "error"
        return "continue"
    
    def run(
        self,
        ticker: str,
        prediction: float,
        conformal_interval: tuple[float, float],
        current_price: float,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the narrator workflow.
        
        Args:
            ticker: Stock ticker symbol
            prediction: Model's predicted next-day return
            conformal_interval: (lower_bound, upper_bound) for conformal prediction interval
            current_price: Current stock price
            query: Optional custom query for retrieval
            
        Returns:
            Dictionary containing the complete workflow result
        """
        # Use simplified workflow if LangGraph not available
        if not LANGGRAPH_AVAILABLE or self.graph is None:
            return self._run_simplified(ticker, prediction, conformal_interval, current_price, query)
        
        # Initialize state
        initial_state: NarratorState = {
            "ticker": ticker,
            "prediction": prediction,
            "conformal_interval": conformal_interval,
            "current_price": current_price,
            "query": query,
            "retrieved_documents": None,
            "narrative": None,
            "citations": None,
            "sentiment": None,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            logger.info(f"Narrator workflow completed for {ticker}")
            
            return {
                "success": True,
                "ticker": ticker,
                "prediction": prediction,
                "conformal_interval": conformal_interval,
                "current_price": current_price,
                "narrative": result.get("narrative"),
                "citations": result.get("citations", []),
                "sentiment": result.get("sentiment"),
                "retrieved_documents": result.get("retrieved_documents"),
                "query": result.get("query"),
                "timestamp": result.get("timestamp"),
                "error": result.get("error"),
                "sources_used": len(result.get("retrieved_documents") or [])
            }
            
        except Exception as e:
            logger.error(f"Error running narrator workflow: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_simplified(
        self,
        ticker: str,
        prediction: float,
        conformal_interval: tuple[float, float],
        current_price: float,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run simplified workflow without LangGraph.
        
        Args:
            ticker: Stock ticker symbol
            prediction: Model's predicted next-day return
            conformal_interval: (lower_bound, upper_bound) for conformal prediction interval
            current_price: Current stock price
            query: Optional custom query for retrieval
            
        Returns:
            Dictionary containing the workflow result
        """
        try:
            # Step 1: Retrieve documents
            if query is None:
                sentiment = "bullish" if prediction > 0 else "bearish"
                query = f"{ticker} stock analysis {sentiment} factors earnings news"
            
            retrieval_results = self.retriever_agent.retrieve_documents(
                query=query,
                ticker=ticker,
                n_results=5
            )
            
            retrieved_docs = retrieval_results["documents"]
            
            # Step 2: Generate narrative
            narrative_result = self.synthesis_agent.generate_narrative(
                prediction=prediction,
                conformal_interval=conformal_interval,
                ticker=ticker,
                retrieved_docs=retrieved_docs,
                current_price=current_price
            )
            
            logger.info(f"Simplified narrator workflow completed for {ticker}")
            
            return {
                "success": True,
                "ticker": ticker,
                "prediction": prediction,
                "conformal_interval": conformal_interval,
                "current_price": current_price,
                "narrative": narrative_result["narrative"],
                "citations": narrative_result["citations"],
                "sentiment": narrative_result["sentiment"],
                "retrieved_documents": retrieved_docs,
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "error": None,
                "simplified_mode": True,
                "sources_used": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error running simplified narrator workflow: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def initialize_vector_store(self, ticker: str = "NFLX") -> None:
        """
        Initialize the vector store with sample documents.
        
        Args:
            ticker: Stock ticker symbol
        """
        try:
            # Prepare documents
            documents, metadatas, ids = self.corpus_manager.prepare_documents_for_vector_store(ticker)
            
            # Clear existing collection
            self.vector_store.clear_collection()
            
            # Add documents
            self.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Vector store initialized with {len(documents)} documents for {ticker}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise