import logging
import time
from datetime import datetime, timezone
from typing import List, Union, Dict

from deepeval.metrics import BaseMetric
from backend.providers.base import LLMProvider
from backend.evaluation.models import (
    TestCase, EvaluationDataset, TestCaseResult, MetricScore,
    EvaluationSummary, EvaluationReport
)
from backend.evaluation.dataset_loader import DatasetLoader
from backend.evaluation.evaluator import TestCaseEvaluator
from backend.evaluation.exceptions import EvaluationError

logger = logging.getLogger("llm_platform.evaluation.engine")

class EvaluationEngine:
    """The main orchestration engine executing benchmark evaluations."""

    def __init__(self, provider: LLMProvider, metrics: List[BaseMetric]):
        self.provider = provider
        self.metrics = metrics

    async def evaluate(self, dataset: Union[str, EvaluationDataset]) -> EvaluationReport:
        """
        Executes evaluation over all test cases in the dataset.
        dataset can be either a parsed EvaluationDataset model or a filepath string.
        """
        if isinstance(dataset, str):
            parsed_dataset = DatasetLoader.load_from_file(dataset)
        elif isinstance(dataset, EvaluationDataset):
            parsed_dataset = dataset
        else:
            raise EvaluationError("Dataset must be an instance of EvaluationDataset or a filepath string")

        logger.info(
            f"Starting evaluation run: dataset={parsed_dataset.id}, test_cases={len(parsed_dataset.test_cases)}, "
            f"provider={self.provider.provider_name}, model={self.provider.model_name}"
        )

        start_time = time.perf_counter()
        results: List[TestCaseResult] = []
        
        passed_count = 0
        failed_count = 0
        error_count = 0
        completed_count = 0

        # Loop through each test case
        for test_case in parsed_dataset.test_cases:
            logger.info(f"Processing test case {test_case.id} in dataset {parsed_dataset.id}")
            case_start = time.perf_counter()
            
            generated_text = None
            scores: List[MetricScore] = []
            success = False
            error_message = None

            try:
                # 1. Generate text using Provider Layer
                provider_response = await self.provider.generate(
                    prompt=test_case.prompt,
                    system_prompt=test_case.metadata.get("system_prompt")
                )
                generated_text = provider_response.text

                # 2. Evaluate using Pluggable Evaluator
                scores = await TestCaseEvaluator.evaluate_case(
                    test_case=test_case,
                    generated_output=generated_text,
                    metrics=self.metrics
                )
                
                # Check if all metrics passed
                case_failed = any(not s.passed for s in scores)
                if case_failed:
                    failed_count += 1
                else:
                    passed_count += 1
                
                success = True
                completed_count += 1

            except Exception as e:
                # Catch any provider or evaluation metric error gracefully to continue evaluation
                error_count += 1
                error_message = str(e)
                logger.error(f"Failed to execute evaluation on test case {test_case.id}: {error_message}")

            case_latency = (time.perf_counter() - case_start) * 1000
            
            results.append(
                TestCaseResult(
                    test_case_id=test_case.id,
                    provider_name=self.provider.provider_name,
                    model_name=self.provider.model_name,
                    prompt=test_case.prompt,
                    generated_output=generated_text,
                    expected_output=test_case.expected_output,
                    scores=scores,
                    success=success,
                    error_message=error_message,
                    latency_ms=case_latency,
                    timestamp=datetime.now(timezone.utc),
                    metadata=test_case.metadata
                )
            )

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Aggregate summary metrics
        summary = self._compute_summary(
            total_cases=len(parsed_dataset.test_cases),
            completed_cases=completed_count,
            passed_cases=passed_count,
            failed_cases=failed_count,
            errors=error_count,
            results=results,
            total_latency=total_latency_ms
        )

        logger.info(
            f"Evaluation run completed: dataset={parsed_dataset.id}, passed={passed_count}, "
            f"failed={failed_count}, errors={error_count}, total_latency={total_latency_ms:.2f}ms"
        )

        return EvaluationReport(
            dataset_id=parsed_dataset.id,
            dataset_name=parsed_dataset.name,
            summary=summary,
            results=results
        )

    def _compute_summary(
        self,
        total_cases: int,
        completed_cases: int,
        passed_cases: int,
        failed_cases: int,
        errors: int,
        results: List[TestCaseResult],
        total_latency: float
    ) -> EvaluationSummary:
        
        avg_latency = 0.0
        if results:
            avg_latency = sum(r.latency_ms for r in results) / len(results)

        # Compute average score per metric name
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        
        for r in results:
            if not r.success:
                continue
            for score in r.scores:
                metric_sums[score.name] = metric_sums.get(score.name, 0.0) + score.score
                metric_counts[score.name] = metric_counts.get(score.name, 0) + 1

        summary_metrics = {}
        for name in metric_sums:
            count = metric_counts[name]
            summary_metrics[name] = metric_sums[name] / count if count > 0 else 0.0

        return EvaluationSummary(
            total_test_cases=total_cases,
            completed_test_cases=completed_cases,
            passed_test_cases=passed_cases,
            failed_test_cases=failed_cases,
            error_count=errors,
            avg_latency_ms=avg_latency,
            total_latency_ms=total_latency,
            summary_metrics=summary_metrics
        )
