import logging
from typing import List
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from backend.evaluation.models import TestCase, MetricScore
from backend.evaluation.exceptions import MetricExecutionError

logger = logging.getLogger("llm_platform.evaluation.evaluator")

class TestCaseEvaluator:
    """Evaluates a single test case's output against a list of DeepEval metrics."""

    @staticmethod
    async def evaluate_case(
        test_case: TestCase,
        generated_output: str,
        metrics: List[BaseMetric]
    ) -> List[MetricScore]:
        """
        Runs the specified metrics against the test case parameters.
        Returns a list of MetricScore models.
        """
        eval_case = LLMTestCase(
            input=test_case.prompt,
            actual_output=generated_output,
            expected_output=test_case.expected_output,
            retrieval_context=test_case.context
        )

        scores: List[MetricScore] = []
        
        for metric in metrics:
            metric_name = getattr(metric, "__name__", type(metric).__name__)
            logger.info(f"Executing metric: {metric_name} for test case {test_case.id}")
            
            try:
                if hasattr(metric, "a_measure"):
                    score_value = await metric.a_measure(eval_case)
                else:
                    score_value = metric.measure(eval_case)
                
                passed = True
                if hasattr(metric, "is_successful"):
                    passed = metric.is_successful()
                elif hasattr(metric, "threshold"):
                    passed = score_value >= metric.threshold

                reason = getattr(metric, "reason", None)

                scores.append(
                    MetricScore(
                        name=metric_name,
                        score=float(score_value),
                        reason=reason,
                        passed=bool(passed)
                    )
                )
            except Exception as e:
                logger.error(f"Metric {metric_name} failed on case {test_case.id}: {str(e)}")
                raise MetricExecutionError(
                    f"Metric {metric_name} execution failed on test case {test_case.id}: {str(e)}"
                )

        return scores
