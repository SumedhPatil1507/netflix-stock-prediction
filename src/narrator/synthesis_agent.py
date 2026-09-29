"""
Synthesis agent for RAG system.
Reads model predictions and retrieved documents to generate plain-English explanations.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not installed. LLM features will be disabled.")

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Agent that synthesizes model predictions with retrieved context into narratives."""
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        """
        Initialize synthesis agent.
        
        Args:
            llm_model: OpenAI model to use
            temperature: Temperature for LLM generation
        """
        self.llm_model = llm_model
        self.temperature = temperature
        
        # Initialize LLM if available
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                streaming=False
            )
            logger.info(f"Synthesis agent initialized with model: {llm_model}")
        else:
            self.llm = None
            logger.warning("Synthesis agent initialized without LLM (will use fallback narratives)")
    
    def generate_narrative(
        self,
        prediction: float,
        conformal_interval: tuple[float, float],
        ticker: str,
        retrieved_docs: List[Dict[str, Any]],
        current_price: float,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a plain-English narrative explaining the model prediction.
        
        Args:
            prediction: Model's predicted next-day return
            conformal_interval: (lower_bound, upper_bound) for conformal prediction interval
            ticker: Stock ticker symbol
            retrieved_docs: List of retrieved documents from retriever agent
            current_price: Current stock price
            additional_context: Additional context (e.g., recent performance, technical indicators)
            
        Returns:
            Dictionary containing the narrative and metadata
        """
        # Determine sentiment
        sentiment = "bullish" if prediction > 0 else "bearish"
        strength = "strongly" if abs(prediction) > 0.02 else "moderately" if abs(prediction) > 0.01 else "slightly"
        
        # Use fallback if LangChain not available
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            logger.info("Using fallback narrative (LangChain not available)")
            return self._generate_fallback_narrative(
                prediction, conformal_interval, ticker, current_price, sentiment, strength
            )
        
        # Format retrieved documents for context
        context = self._format_documents_for_context(retrieved_docs)
        
        # Create system prompt
        system_prompt = """You are a financial market analyst AI that explains stock price predictions in plain, clear English. 
Your task is to synthesize model predictions with relevant financial news and earnings call information to create compelling, citation-backed narratives.

Follow these guidelines:
1. Start with a clear, actionable summary of the prediction (bullish/bearish)
2. Explain the key factors driving the prediction based on retrieved documents
3. Include specific citations from the provided documents using [Source: Document Title, Date]
4. Quantify the prediction with percentage returns and confidence intervals
5. Mention risks and uncertainties when relevant
6. Keep the tone professional but accessible
7. Be honest about limitations and uncertainties

Structure your response with:
- Executive Summary (2-3 sentences)
- Key Drivers (3-5 bullet points with citations)
- Risk Factors (2-3 bullet points)
- Conclusion (1-2 sentences)"""
        
        # Create user prompt
        user_prompt = f"""Generate a market narrative for {ticker} based on the following information:

MODEL PREDICTION:
- Predicted next-day return: {prediction:.4f} ({prediction*100:.2f}%)
- Sentiment: {sentiment} ({strength})
- Conformal prediction interval: [{conformal_interval[0]:.4f}, {conformal_interval[1]:.4f}] ([{conformal_interval[0]*100:.2f}%, {conformal_interval[1]*100:.2f}%])
- Current price: ${current_price:.2f}

RETRIEVED FINANCIAL CONTEXT:
{context}

ADDITIONAL CONTEXT:
{self._format_additional_context(additional_context)}

Please provide a clear, well-structured narrative that explains why the model is {sentiment} on {ticker}, backed by specific citations from the retrieved documents."""
        
        try:
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            narrative = response.content
            
            # Extract citations
            citations = self._extract_citations(narrative, retrieved_docs)
            
            logger.info(f"Generated narrative for {ticker}: {sentiment} prediction")
            
            return {
                "ticker": ticker,
                "prediction": prediction,
                "conformal_interval": conformal_interval,
                "sentiment": sentiment,
                "strength": strength,
                "narrative": narrative,
                "citations": citations,
                "sources_used": len(retrieved_docs),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            # Return fallback narrative
            return self._generate_fallback_narrative(
                prediction, conformal_interval, ticker, current_price, sentiment, strength
            )
    
    def _format_documents_for_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for the LLM context."""
        if not docs:
            return "No relevant documents retrieved."
        
        formatted = ""
        for i, doc in enumerate(docs, 1):
            formatted += f"\nDocument {i}:\n"
            formatted += f"Title: {doc['metadata'].get('title', 'N/A')}\n"
            formatted += f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            formatted += f"Date: {doc['metadata'].get('date', 'Unknown')}\n"
            formatted += f"Content: {doc['text']}\n"
            formatted += "-" * 80 + "\n"
        
        return formatted
    
    def _format_additional_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format additional context for the LLM."""
        if not context:
            return "No additional context provided."
        
        formatted = ""
        for key, value in context.items():
            formatted += f"- {key}: {value}\n"
        
        return formatted if formatted else "No additional context provided."
    
    def _extract_citations(
        self,
        narrative: str,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and format citations from the narrative."""
        citations = []
        
        # Simple extraction based on document titles mentioned in narrative
        for doc in docs:
            title = doc['metadata'].get('title', '')
            if title and title.lower() in narrative.lower():
                citations.append({
                    "title": title,
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "date": doc['metadata'].get('date', 'Unknown'),
                    "url": doc['metadata'].get('url', '')
                })
        
        return citations
    
    def _generate_fallback_narrative(
        self,
        prediction: float,
        conformal_interval: tuple[float, float],
        ticker: str,
        current_price: float,
        sentiment: str,
        strength: str
    ) -> Dict[str, Any]:
        """Generate a fallback narrative if LLM fails."""
        narrative = f"""Executive Summary:
The model predicts a {strength} {sentiment} outlook for {ticker} with a predicted next-day return of {prediction*100:.2f}%. 
The conformal prediction interval suggests the return will likely fall between {conformal_interval[0]*100:.2f}% and {conformal_interval[1]*100:.2f}%.

Key Drivers:
- Current price: ${current_price:.2f}
- Model prediction based on technical indicators and historical patterns
- Prediction confidence interval: {conformal_interval[1] - conformal_interval[0]:.4f}

Risk Factors:
- Market volatility and unexpected news events
- Model limitations and potential overfitting
- External factors not captured in the analysis

Conclusion:
The model indicates a {strength} {sentiment} sentiment for {ticker}, but as with all predictions, this should be considered as one input among many in investment decisions."""

        return {
            "ticker": ticker,
            "prediction": prediction,
            "conformal_interval": conformal_interval,
            "sentiment": sentiment,
            "strength": strength,
            "narrative": narrative,
            "citations": [],
            "sources_used": 0,
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        }
    
    def summarize_findings(
        self,
        narratives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize findings from multiple narratives.
        
        Args:
            narratives: List of narrative dictionaries
            
        Returns:
            Dictionary containing summary
        """
        if not narratives:
            return {"error": "No narratives to summarize"}
        
        # Count sentiments
        bullish_count = sum(1 for n in narratives if n["sentiment"] == "bullish")
        bearish_count = sum(1 for n in narratives if n["sentiment"] == "bearish")
        
        # Average prediction
        avg_prediction = sum(n["prediction"] for n in narratives) / len(narratives)
        
        summary = {
            "total_narratives": len(narratives),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "average_prediction": avg_prediction,
            "consensus": "bullish" if bullish_count > bearish_count else "bearish",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Summary: {bullish_count} bullish, {bearish_count} bearish out of {len(narratives)} narratives")
        
        return summary