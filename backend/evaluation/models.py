from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class TestCase(BaseModel):
    __test__ = False
    id: str = Field(..., description="Unique identifier for the evaluation test case")
    prompt: str = Field(..., description="Prompt sent to the model under test")
    expected_output: Optional[str] = Field(None, description="The gold standard or expected output response")
    context: Optional[List[str]] = Field(None, description="Optional retrieval context/documents for faithfulness/hallucination checks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata variables for the test case")


class EvaluationDataset(BaseModel):
    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Human-readable dataset name")
    test_cases: List[TestCase] = Field(default_factory=list, description="Array of strongly-typed evaluation test cases")


class MetricScore(BaseModel):
    name: str = Field(..., description="Name of the evaluation metric (e.g. Exact Match, Faithfulness)")
    score: float = Field(..., description="Score value (typically between 0 and 1)")
    reason: Optional[str] = Field(None, description="Qualitative feedback or reason explaining the score")
    passed: bool = Field(..., description="Indicates if the test case met the metric's threshold")


class TestCaseResult(BaseModel):
    __test__ = False
    test_case_id: str = Field(..., description="Reference test case ID")
    provider_name: str = Field(..., description="Name of the LLM provider used")
    model_name: str = Field(..., description="Model ID used")
    prompt: str = Field(..., description="The input prompt executed")
    generated_output: Optional[str] = Field(None, description="The text completion returned by the provider")
    expected_output: Optional[str] = Field(None, description="Expected response text, if any")
    scores: List[MetricScore] = Field(default_factory=list, description="Computed scores across all evaluated metrics")
    success: bool = Field(..., description="True if the test case generated and evaluated successfully, False if any error occurred")
    error_message: Optional[str] = Field(None, description="Error message details if success is False")
    latency_ms: float = Field(..., description="Response latency of provider + evaluator in milliseconds")
    timestamp: datetime = Field(..., description="Completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata carried over from the test case")


class EvaluationSummary(BaseModel):
    total_test_cases: int = Field(..., description="Total test cases parsed")
    completed_test_cases: int = Field(..., description="Number of successfully generated and graded cases")
    passed_test_cases: int = Field(..., description="Number of cases meeting success criteria for all metrics")
    failed_test_cases: int = Field(..., description="Number of cases failing at least one metric")
    error_count: int = Field(..., description="Number of test cases encountering hard execution errors")
    avg_latency_ms: float = Field(..., description="Average latency across all execution blocks")
    total_latency_ms: float = Field(..., description="Total elapsed engine running time in milliseconds")
    summary_metrics: Dict[str, float] = Field(default_factory=dict, description="Aggregated average scores for each metric type")


class EvaluationReport(BaseModel):
    dataset_id: str = Field(..., description="ID of the parsed dataset")
    dataset_name: str = Field(..., description="Name of the parsed dataset")
    summary: EvaluationSummary = Field(..., description="Consolidated latency and score statistical properties")
    results: List[TestCaseResult] = Field(default_factory=list, description="Array of detailed results per test case")
