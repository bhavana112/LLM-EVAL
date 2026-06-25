import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.reporting.models import ExperimentReport, RegressionComparisonReport
from backend.reporting.exceptions import IncompatibleExperimentsError

logger = logging.getLogger("llm_platform.reporting.regression_detector")

class RegressionDetector:
    """Compares experiment reports to identify performance regressions, improvements, or latency drifts."""

    def __init__(self, tolerance: float = 0.01):
        """
        tolerance: Minimum difference threshold to declare a regression or improvement (default 1%).
        """
        self.tolerance = tolerance

    def compare(self, current: ExperimentReport, previous: ExperimentReport) -> RegressionComparisonReport:
        """
        Compares the current report against a previous report.
        Raises IncompatibleExperimentsError if the datasets are different.
        """
        logger.info(f"Comparing current experiment '{current.experiment_id}' with previous '{previous.experiment_id}'")

        if current.dataset_name != previous.dataset_name:
            raise IncompatibleExperimentsError(
                f"Experiments are incompatible: Current dataset is '{current.dataset_name}' "
                f"but previous dataset is '{previous.dataset_name}'."
            )

        # 1. Overall deltas
        score_diff = current.overall_score - previous.overall_score
        pass_rate_diff = current.pass_rate - previous.pass_rate
        latency_diff = current.average_latency - previous.average_latency

        # 2. Metric-specific deltas
        metric_diffs: Dict[str, float] = {}
        curr_metrics = current.average_score_for_every_evaluation_metric
        prev_metrics = previous.average_score_for_every_evaluation_metric

        all_metrics = set(curr_metrics.keys()).union(set(prev_metrics.keys()))
        change_summary: List[str] = []

        regressed_metrics = []
        improved_metrics = []

        for metric in all_metrics:
            if metric in curr_metrics and metric in prev_metrics:
                diff = curr_metrics[metric] - prev_metrics[metric]
                metric_diffs[metric] = diff
                
                if diff < -self.tolerance:
                    regressed_metrics.append((metric, diff))
                    change_summary.append(f"Metric '{metric}' regressed by {diff:+.2f} (from {prev_metrics[metric]:.2f} to {curr_metrics[metric]:.2f})")
                elif diff > self.tolerance:
                    improved_metrics.append((metric, diff))
                    change_summary.append(f"Metric '{metric}' improved by {diff:+.2f} (from {prev_metrics[metric]:.2f} to {curr_metrics[metric]:.2f})")
            elif metric in curr_metrics:
                change_summary.append(f"Metric '{metric}' added in current run (average score: {curr_metrics[metric]:.2f})")
            else:
                change_summary.append(f"Metric '{metric}' missing in current run (was: {prev_metrics[metric]:.2f})")

        regression_status = False
        improvement_status = False

        if score_diff < -self.tolerance:
            regression_status = True
            change_summary.append(f"Overall evaluation score regressed by {score_diff:+.2f}")
        elif score_diff > self.tolerance:
            improvement_status = True
            change_summary.append(f"Overall evaluation score improved by {score_diff:+.2f}")

        if regressed_metrics:
            regression_status = True

        if abs(latency_diff) > 50.0:  # Highlight latency changes larger than 50ms
            if latency_diff > 0:
                change_summary.append(f"Average response latency increased by {latency_diff:+.2f}ms")
            else:
                change_summary.append(f"Average response latency decreased by {latency_diff:+.2f}ms")

        if abs(pass_rate_diff) > 0.001:
            change_summary.append(f"Test case pass rate changed by {pass_rate_diff:+.2%}")

        if regression_status:
            verdict = "Worse"
        elif improvement_status:
            verdict = "Better"
        else:
            verdict = "Approximately the same"

        return RegressionComparisonReport(
            previous_experiment_id=previous.experiment_id,
            current_experiment_id=current.experiment_id,
            comparison_timestamp=datetime.now(timezone.utc),
            score_difference=score_diff,
            metric_differences=metric_diffs,
            latency_difference=latency_diff,
            pass_rate_difference=pass_rate_diff,
            regression_status=regression_status,
            improvement_status=improvement_status,
            change_summary=change_summary,
            performance_verdict=verdict
        )
