from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseMetricEvaluator(ABC):
    """Abstract interface for evaluating generation quality/metrics."""

    @abstractmethod
    async def evaluate(self, generated_text: str, expected_text: Optional[str] = None) -> Dict[str, float]:
        """Compute evaluation metrics."""
        pass


class SimpleMetricEvaluator(BaseMetricEvaluator):
    """Mock/Simple evaluation metrics (exact match, basic word overlap/similarity)."""

    async def evaluate(self, generated_text: str, expected_text: Optional[str] = None) -> Dict[str, float]:
        metrics = {}
        
        # Word count metrics
        metrics["word_count"] = float(len(generated_text.split()))
        metrics["char_count"] = float(len(generated_text))

        if expected_text:
            # Exact Match metric
            metrics["exact_match"] = 1.0 if generated_text.strip().lower() == expected_text.strip().lower() else 0.0
            
            # Simple Jaccard similarity as placeholder for semantic metrics
            gen_words = set(generated_text.lower().split())
            exp_words = set(expected_text.lower().split())
            if gen_words or exp_words:
                intersection = gen_words.intersection(exp_words)
                union = gen_words.union(exp_words)
                metrics["jaccard_similarity"] = len(intersection) / len(union)
            else:
                metrics["jaccard_similarity"] = 1.0
                
        return metrics


_evaluator_instance: Optional[BaseMetricEvaluator] = None

def get_metric_evaluator() -> BaseMetricEvaluator:
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = SimpleMetricEvaluator()
    return _evaluator_instance
