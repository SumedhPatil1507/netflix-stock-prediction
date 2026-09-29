"""
RAGAS-based evaluation script for faithfulness scoring.
Evaluates the quality and faithfulness of generated narratives against retrieved sources.
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import pandas as pd

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.dataset import Dataset
    from langchain_core.documents import Document
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not installed. Evaluation features will be disabled.")

logger = logging.getLogger(__name__)


class NarrativeEvaluator:
    """Evaluator for AI narratives using RAGAS metrics."""
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        enable_evaluation: bool = True
    ):
        """
        Initialize narrative evaluator.
        
        Args:
            metrics: List of metrics to use (faithfulness, answer_relevancy, context_precision)
            enable_evaluation: Whether to enable evaluation
        """
        self.enable_evaluation = enable_evaluation and RAGAS_AVAILABLE
        
        if self.enable_evaluation:
            self.metrics = metrics or ["faithfulness", "answer_relevancy"]
            self.metric_objects = []
            
            for metric_name in self.metrics:
                if metric_name == "faithfulness":
                    self.metric_objects.append(faithfulness)
                elif metric_name == "answer_relevancy":
                    self.metric_objects.append(answer_relevancy)
                elif metric_name == "context_precision":
                    self.metric_objects.append(context_precision)
            
            logger.info(f"Narrative evaluator initialized with metrics: {self.metrics}")
        else:
            self.metrics = []
            self.metric_objects = []
            logger.info("Narrative evaluation disabled")
    
    def evaluate_narrative(
        self,
        narrative: str,
        retrieved_docs: List[Dict[str, Any]],
        query: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single narrative against retrieved documents.
        
        Args:
            narrative: Generated narrative text
            retrieved_docs: List of retrieved documents
            query: Original query used for retrieval
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary containing evaluation scores
        """
        if not self.enable_evaluation:
            return {
                "error": "Evaluation disabled or RAGAS not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Prepare data for RAGAS
            contexts = [
                [doc["text"] for doc in retrieved_docs]
            ]
            
            questions = [query]
            answers = [narrative]
            ground_truths = [ground_truth] if ground_truth else None
            
            # Create dataset
            dataset_dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts
            }
            
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metric_objects
            )
            
            # Convert results to dictionary
            scores = results.to_pandas().to_dict('records')[0] if len(results.to_pandas()) > 0 else {}
            
            logger.info(f"Evaluation completed for narrative: {scores}")
            
            return {
                "scores": scores,
                "metrics_used": self.metrics,
                "narrative_length": len(narrative),
                "num_contexts": len(retrieved_docs),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating narrative: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_batch(
        self,
        narratives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of narratives.
        
        Args:
            narratives: List of narrative dictionaries with keys:
                - narrative: Generated narrative text
                - retrieved_docs: List of retrieved documents
                - query: Original query
                - ground_truth: Optional ground truth
                
        Returns:
            Dictionary containing batch evaluation results
        """
        if not self.enable_evaluation:
            return {
                "error": "Evaluation disabled or RAGAS not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Prepare batch data
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            for item in narratives:
                questions.append(item.get("query", ""))
                answers.append(item.get("narrative", ""))
                contexts.append([doc["text"] for doc in item.get("retrieved_docs", [])])
                if item.get("ground_truth"):
                    ground_truths.append(item["ground_truth"])
            
            # Create dataset
            dataset_dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts
            }
            
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metric_objects
            )
            
            # Convert results to dictionary
            results_df = results.to_pandas()
            
            # Calculate aggregate statistics
            aggregate_scores = {}
            for metric in self.metrics:
                if metric in results_df.columns:
                    aggregate_scores[f"{metric}_mean"] = results_df[metric].mean()
                    aggregate_scores[f"{metric}_std"] = results_df[metric].std()
                    aggregate_scores[f"{metric}_min"] = results_df[metric].min()
                    aggregate_scores[f"{metric}_max"] = results_df[metric].max()
            
            logger.info(f"Batch evaluation completed for {len(narratives)} narratives")
            
            return {
                "aggregate_scores": aggregate_scores,
                "individual_scores": results_df.to_dict('records'),
                "metrics_used": self.metrics,
                "num_narratives": len(narratives),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_faithfulness_score(
        self,
        narrative: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a simple faithfulness score based on citation overlap.
        
        Args:
            narrative: Generated narrative text
            retrieved_docs: List of retrieved documents
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract key terms from narrative
        narrative_words = set(narrative.lower().split())
        
        # Calculate overlap with retrieved documents
        total_overlap = 0
        for doc in retrieved_docs:
            doc_words = set(doc["text"].lower().split())
            overlap = len(narrative_words & doc_words)
            total_overlap += overlap
        
        # Normalize by number of documents
        avg_overlap = total_overlap / len(retrieved_docs)
        
        # Normalize by narrative length
        faithfulness = min(avg_overlap / len(narrative_words), 1.0) if narrative_words else 0.0
        
        return faithfulness
    
    def generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_narrative or evaluate_batch
            output_path: Optional path to save the report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("AI NARRATIVE EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {evaluation_results.get('timestamp', 'Unknown')}")
        report.append("")
        
        if "error" in evaluation_results:
            report.append(f"ERROR: {evaluation_results['error']}")
            return "\n".join(report)
        
        if "aggregate_scores" in evaluation_results:
            # Batch evaluation report
            report.append("BATCH EVALUATION RESULTS")
            report.append("-" * 80)
            report.append(f"Number of narratives evaluated: {evaluation_results.get('num_narratives', 0)}")
            report.append(f"Metrics used: {', '.join(evaluation_results.get('metrics_used', []))}")
            report.append("")
            
            report.append("AGGREGATE SCORES:")
            for metric, score in evaluation_results.get('aggregate_scores', {}).items():
                report.append(f"  {metric}: {score:.4f}")
            
            report.append("")
            report.append("INDIVIDUAL SCORES:")
            for i, score_dict in enumerate(evaluation_results.get('individual_scores', [])):
                report.append(f"  Narrative {i+1}:")
                for metric, score in score_dict.items():
                    if isinstance(score, (int, float)):
                        report.append(f"    {metric}: {score:.4f}")
        
        elif "scores" in evaluation_results:
            # Single evaluation report
            report.append("SINGLE NARRATIVE EVALUATION")
            report.append("-" * 80)
            report.append(f"Metrics used: {', '.join(evaluation_results.get('metrics_used', []))}")
            report.append(f"Narrative length: {evaluation_results.get('narrative_length', 0)} characters")
            report.append(f"Number of contexts: {evaluation_results.get('num_contexts', 0)}")
            report.append("")
            
            report.append("SCORES:")
            for metric, score in evaluation_results.get('scores', {}).items():
                if isinstance(score, (int, float)):
                    report.append(f"  {metric}: {score:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Evaluation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving evaluation report: {e}")
        
        return report_text
    
    def save_evaluation_results(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str = "outputs/evaluation_results.json"
    ) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            evaluation_results: Results from evaluation
            output_path: Path to save the results
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


def run_evaluation_pipeline(
    narratives: List[Dict[str, Any]],
    output_dir: str = "outputs/evaluations"
) -> Dict[str, Any]:
    """
    Run a complete evaluation pipeline on a batch of narratives.
    
    Args:
        narratives: List of narrative dictionaries
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = NarrativeEvaluator()
    
    # Run batch evaluation
    batch_results = evaluator.evaluate_batch(narratives)
    
    # Generate report
    report = evaluator.generate_evaluation_report(batch_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
    
    evaluator.save_evaluation_results(batch_results, results_path)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation pipeline completed. Results saved to {output_dir}")
    
    return {
        "batch_results": batch_results,
        "report": report,
        "results_path": results_path,
        "report_path": report_path
    }