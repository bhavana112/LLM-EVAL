import pytest
import os
import shutil
import tempfile
from datetime import datetime, timezone

from backend.experiments.models import Experiment, EvaluationResultEntry
from backend.reporting import (
    ReportGenerator, RegressionDetector, ExperimentReport, RegressionComparisonReport,
    IncompatibleExperimentsError, ReportGenerationError
)

@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)


def test_report_generation_from_experiment():
    generator = ReportGenerator()
    
    results = [
        EvaluationResultEntry(
            test_case_id="tc-1",
            prompt="Capital of UK?",
            generated_output="London",
            expected_output="London",
            scores={"Exact Match": 1.0, "Jaccard": 1.0},
            latency_ms=50.0,
            timestamp=datetime.now(timezone.utc),
            passed=True
        ),
        EvaluationResultEntry(
            test_case_id="tc-2",
            prompt="Capital of France?",
            generated_output="Lyon",
            expected_output="Paris",
            scores={"Exact Match": 0.0, "Jaccard": 0.0},
            latency_ms=150.0,
            timestamp=datetime.now(timezone.utc),
            passed=False
        )
    ]
    
    experiment = Experiment(
        experiment_id="exp_run_123",
        timestamp=datetime.now(timezone.utc),
        dataset_name="Geography Q&A",
        dataset_version="v1",
        provider="openai",
        model="gpt-4o",
        evaluation_configuration={"temperature": 0.0},
        evaluation_results=results,
        evaluation_metrics={"Exact Match": 0.5, "Jaccard": 0.5},
        average_latency=100.0,
        total_number_of_test_cases=2,
        passed_test_cases=1,
        failed_test_cases=1
    )
    
    report = generator.generate_report(experiment)
    
    assert isinstance(report, ExperimentReport)
    assert report.experiment_id == "exp_run_123"
    assert report.total_number_of_test_cases == 2
    assert report.successful_evaluations == 2
    assert report.failed_evaluations == 0
    assert report.overall_score == 0.5
    assert report.average_score_for_every_evaluation_metric["Exact Match"] == 0.5
    assert report.pass_rate == 0.5
    assert report.average_latency == 100.0
    assert report.minimum_latency == 50.0
    assert report.maximum_latency == 150.0
    assert report.total_execution_time_ms == 200.0
    assert "overall score: 50.00%" in report.evaluation_summary.lower()


def test_report_generation_empty():
    generator = ReportGenerator()
    
    experiment = Experiment(
        experiment_id="exp_empty",
        timestamp=datetime.now(timezone.utc),
        dataset_name="Geography Q&A",
        provider="openai",
        model="gpt-4o",
        evaluation_configuration={},
        evaluation_results=[],
        evaluation_metrics={},
        average_latency=0.0,
        total_number_of_test_cases=0,
        passed_test_cases=0,
        failed_test_cases=0
    )
    
    report = generator.generate_report(experiment)
    assert report.total_number_of_test_cases == 0
    assert report.overall_score == 0.0
    assert "no evaluation cases" in report.evaluation_summary.lower()


def test_save_and_load_report(temp_dir):
    generator = ReportGenerator(base_dir=temp_dir)
    
    exp_dir = os.path.join(temp_dir, "exp_1")
    os.makedirs(exp_dir)
    
    report = ExperimentReport(
        experiment_id="exp_1",
        experiment_timestamp=datetime.now(timezone.utc),
        provider="mock",
        model="mock",
        dataset_name="Dataset",
        total_number_of_test_cases=5,
        successful_evaluations=5,
        failed_evaluations=0,
        overall_score=0.8,
        average_score_for_every_evaluation_metric={"Exact Match": 0.8},
        pass_rate=0.8,
        average_latency=10.0,
        minimum_latency=5.0,
        maximum_latency=15.0,
        total_execution_time_ms=50.0,
        evaluation_summary="Summary"
    )
    
    generator.save_report(report)
    assert os.path.exists(os.path.join(exp_dir, "report.json"))
    
    loaded = generator.load_report("exp_1")
    assert loaded is not None
    assert loaded.experiment_id == "exp_1"
    assert loaded.overall_score == 0.8


def test_save_report_non_existent_folder_raises_error(temp_dir):
    generator = ReportGenerator(base_dir=temp_dir)
    report = ExperimentReport(
        experiment_id="exp_non_existent",
        experiment_timestamp=datetime.now(timezone.utc),
        provider="mock",
        model="mock",
        dataset_name="Dataset",
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
        evaluation_summary="Summary"
    )
    with pytest.raises(ReportGenerationError):
        generator.save_report(report)


def test_regression_detector_incompatible_datasets():
    detector = RegressionDetector()
    report1 = ExperimentReport(
        experiment_id="exp_1", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=0,
        successful_evaluations=0, failed_evaluations=0, overall_score=0.0, average_score_for_every_evaluation_metric={},
        pass_rate=0.0, average_latency=0.0, minimum_latency=0.0, maximum_latency=0.0, total_execution_time_ms=0.0,
        evaluation_summary="Summary"
    )
    report2 = ExperimentReport(
        experiment_id="exp_2", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset B", total_number_of_test_cases=0,
        successful_evaluations=0, failed_evaluations=0, overall_score=0.0, average_score_for_every_evaluation_metric={},
        pass_rate=0.0, average_latency=0.0, minimum_latency=0.0, maximum_latency=0.0, total_execution_time_ms=0.0,
        evaluation_summary="Summary"
    )
    with pytest.raises(IncompatibleExperimentsError):
        detector.compare(report1, report2)


def test_regression_detector_no_change():
    detector = RegressionDetector(tolerance=0.01)
    
    report_prev = ExperimentReport(
        experiment_id="exp_prev", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=1,
        successful_evaluations=1, failed_evaluations=0, overall_score=0.80, 
        average_score_for_every_evaluation_metric={"Exact Match": 0.80},
        pass_rate=0.80, average_latency=100.0, minimum_latency=100.0, maximum_latency=100.0, total_execution_time_ms=100.0,
        evaluation_summary="Summary"
    )
    report_curr = ExperimentReport(
        experiment_id="exp_curr", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=1,
        successful_evaluations=1, failed_evaluations=0, overall_score=0.805,
        average_score_for_every_evaluation_metric={"Exact Match": 0.805},
        pass_rate=0.805, average_latency=102.0, minimum_latency=102.0, maximum_latency=102.0, total_execution_time_ms=102.0,
        evaluation_summary="Summary"
    )
    
    comparison = detector.compare(report_curr, report_prev)
    assert comparison.regression_status is False
    assert comparison.improvement_status is False
    assert comparison.performance_verdict == "Approximately the same"
    assert abs(comparison.score_difference - 0.005) < 0.0001


def test_regression_detector_improvement():
    detector = RegressionDetector(tolerance=0.01)
    
    report_prev = ExperimentReport(
        experiment_id="exp_prev", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=10,
        successful_evaluations=10, failed_evaluations=0, overall_score=0.50, 
        average_score_for_every_evaluation_metric={"Exact Match": 0.50},
        pass_rate=0.50, average_latency=100.0, minimum_latency=100.0, maximum_latency=100.0, total_execution_time_ms=1000.0,
        evaluation_summary="Summary"
    )
    report_curr = ExperimentReport(
        experiment_id="exp_curr", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=10,
        successful_evaluations=10, failed_evaluations=0, overall_score=0.55,
        average_score_for_every_evaluation_metric={"Exact Match": 0.55},
        pass_rate=0.55, average_latency=80.0, minimum_latency=80.0, maximum_latency=80.0, total_execution_time_ms=800.0,
        evaluation_summary="Summary"
    )
    
    comparison = detector.compare(report_curr, report_prev)
    assert comparison.regression_status is False
    assert comparison.improvement_status is True
    assert comparison.performance_verdict == "Better"
    assert comparison.score_difference == pytest.approx(0.05)


def test_regression_detector_regression():
    detector = RegressionDetector(tolerance=0.01)
    
    report_prev = ExperimentReport(
        experiment_id="exp_prev", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=10,
        successful_evaluations=10, failed_evaluations=0, overall_score=0.70, 
        average_score_for_every_evaluation_metric={"Exact Match": 0.70},
        pass_rate=0.70, average_latency=100.0, minimum_latency=100.0, maximum_latency=100.0, total_execution_time_ms=1000.0,
        evaluation_summary="Summary"
    )
    report_curr = ExperimentReport(
        experiment_id="exp_curr", experiment_timestamp=datetime.now(timezone.utc),
        provider="mock", model="mock", dataset_name="Dataset A", total_number_of_test_cases=10,
        successful_evaluations=10, failed_evaluations=0, overall_score=0.65,
        average_score_for_every_evaluation_metric={"Exact Match": 0.65},
        pass_rate=0.65, average_latency=160.0, minimum_latency=160.0, maximum_latency=160.0, total_execution_time_ms=1600.0,
        evaluation_summary="Summary"
    )
    
    comparison = detector.compare(report_curr, report_prev)
    assert comparison.regression_status is True
    assert comparison.improvement_status is False
    assert comparison.performance_verdict == "Worse"
    assert comparison.score_difference == pytest.approx(-0.05)
    # Average response latency was 100ms and became 160ms, which is 60ms difference (>50ms threshold)
    any_latency_note = any("latency increased" in note for note in comparison.change_summary)
    assert any_latency_note is True
