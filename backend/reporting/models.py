from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class ExperimentReport(BaseModel):
    __test__ = False
    experiment_id: str = Field(..., description="Unique experiment ID reference")
    experiment_timestamp: datetime = Field(..., description="UTC timestamp of the experiment completion")
    provider: str = Field(..., description="LLM provider name used")
    model: str = Field(..., description="LLM model name used")
    dataset_name: str = Field(..., description="Dataset name evaluated")
    total_number_of_test_cases: int = Field(..., description="Total test cases evaluated")
    successful_evaluations: int = Field(..., description="Number of evaluations completing successfully")
    failed_evaluations: int = Field(..., description="Number of evaluations failing metrics or encountering errors")
    overall_score: float = Field(..., description="Calculated overall score (average of successful metric scores)")
    average_score_for_every_evaluation_metric: Dict[str, float] = Field(
        default_factory=dict, 
        description="Average score mapped per evaluation metric"
    )
    pass_rate: float = Field(..., description="Pass rate ratio (0.0 to 1.0) of successful test cases")
    average_latency: float = Field(..., description="Average latency in milliseconds")
    minimum_latency: float = Field(..., description="Minimum latency in milliseconds")
    maximum_latency: float = Field(..., description="Maximum latency in milliseconds")
    total_execution_time_ms: float = Field(..., description="Total experiment execution time in milliseconds")
    evaluation_summary: str = Field(..., description="Human-readable text summarizing the run statistics")


class RegressionComparisonReport(BaseModel):
    __test__ = False
    previous_experiment_id: str = Field(..., description="ID of the previous benchmark experiment")
    current_experiment_id: str = Field(..., description="ID of the current benchmark experiment")
    comparison_timestamp: datetime = Field(..., description="Timestamp of the comparison analysis")
    score_difference: float = Field(..., description="Difference in overall score (current - previous)")
    metric_differences: Dict[str, float] = Field(
        default_factory=dict, 
        description="Difference in individual metric averages (current - previous)"
    )
    latency_difference: float = Field(..., description="Difference in average latency in milliseconds (current - previous)")
    pass_rate_difference: float = Field(..., description="Difference in pass rate ratio (current - previous)")
    regression_status: bool = Field(..., description="True if any metric score regressed beyond tolerance")
    improvement_status: bool = Field(..., description="True if overall score or metrics improved")
    change_summary: List[str] = Field(default_factory=list, description="Textual descriptions of all detected regressions/improvements")
    performance_verdict: str = Field(..., description="Verdicts: 'Better', 'Worse', or 'Approximately the same'")
