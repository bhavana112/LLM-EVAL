import logging
import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict

from backend.experiments.models import Experiment
from backend.reporting.models import ExperimentReport
from backend.reporting.exceptions import ReportGenerationError

logger = logging.getLogger("llm_platform.reporting.generator")

class ReportGenerator:
    """Generates and stores evaluation reports from completed experiments."""

    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir

    def generate_report(self, experiment: Experiment) -> ExperimentReport:
        """
        Processes a completed experiment and computes overall evaluation statistics.
        Returns a strongly-typed ExperimentReport model.
        """
        logger.info(f"Generating evaluation report for experiment: {experiment.experiment_id}")
        
        results = experiment.evaluation_results
        if not results:
            return ExperimentReport(
                experiment_id=experiment.experiment_id,
                experiment_timestamp=experiment.timestamp,
                provider=experiment.provider,
                model=experiment.model,
                dataset_name=experiment.dataset_name,
                total_number_of_test_cases=0,
                successful_evaluations=0,
                failed_evaluations=0,
                overall_score=0.0,
                average_score_for_every_evaluation_metric={},
                pass_rate=0.0,
                average_latency=0.0,
                minimum_latency=0.0,
                maximum_latency=0.0,
                total_execution_time_ms=0.0,
                evaluation_summary="No evaluation cases were run in this experiment."
            )

        total_cases = len(results)
        successful_evals = sum(1 for r in results if r.success)
        failed_evals = total_cases - successful_evals

        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / total_cases if total_cases > 0 else 0.0
        min_latency = min(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        total_exec_time = sum(latencies)

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        total_scores_sum = 0.0
        total_scores_count = 0

        for r in results:
            if not r.success:
                continue
            for metric_name, score_val in r.scores.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + score_val
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
                total_scores_sum += score_val
                total_scores_count += 1

        avg_metrics: Dict[str, float] = {}
        for name in metric_sums:
            avg_metrics[name] = metric_sums[name] / metric_counts[name]

        overall_score = total_scores_sum / total_scores_count if total_scores_count > 0 else 0.0

        passed_cases = sum(1 for r in results if r.passed)
        pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0

        summary_text = (
            f"Experiment '{experiment.experiment_id}' evaluated {total_cases} test cases "
            f"on model '{experiment.model}' via '{experiment.provider}'. "
            f"Overall score: {overall_score:.2%}, Pass rate: {pass_rate:.2%}, "
            f"Average latency: {avg_latency:.2f}ms."
        )

        return ExperimentReport(
            experiment_id=experiment.experiment_id,
            experiment_timestamp=experiment.timestamp,
            provider=experiment.provider,
            model=experiment.model,
            dataset_name=experiment.dataset_name,
            total_number_of_test_cases=total_cases,
            successful_evaluations=successful_evals,
            failed_evaluations=failed_evals,
            overall_score=overall_score,
            average_score_for_every_evaluation_metric=avg_metrics,
            pass_rate=pass_rate,
            average_latency=avg_latency,
            minimum_latency=min_latency,
            maximum_latency=max_latency,
            total_execution_time_ms=total_exec_time,
            evaluation_summary=summary_text
        )

    def save_report(self, report: ExperimentReport) -> None:
        """
        Saves the generated report to disk in the corresponding experiment folder.
        """
        exp_dir = os.path.join(self.base_dir, report.experiment_id)
        file_path = os.path.join(exp_dir, "report.json")

        if not os.path.exists(exp_dir):
            raise ReportGenerationError(f"Experiment folder '{exp_dir}' does not exist. Save the experiment first.")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            logger.info(f"Report JSON successfully saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write report JSON to {file_path}: {str(e)}")
            raise ReportGenerationError(f"Failed to save report to filesystem: {str(e)}", details=str(e))

    def load_report(self, experiment_id: str) -> Optional[ExperimentReport]:
        """
        Loads the report.json file of an experiment if it exists.
        """
        file_path = os.path.join(self.base_dir, experiment_id, "report.json")
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ExperimentReport(**data)
        except Exception as e:
            logger.error(f"Failed to read report JSON for '{experiment_id}': {str(e)}")
            return None
