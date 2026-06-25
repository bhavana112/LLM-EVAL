from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# --- Original/Scaffold Models ---

class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    provider_name: str
    model_name: str
    dataset_id: str
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    system_instruction: Optional[str] = None


class ExperimentResultEntry(BaseModel):
    prompt_id: str
    prompt: str
    expected_output: Optional[str] = None
    generated_output: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float
    usage: Dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentRun(BaseModel):
    __test__ = False
    id: str
    config: ExperimentConfig
    results: List[ExperimentResultEntry] = Field(default_factory=list)
    summary_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # "pending", "running", "completed", "failed"


# --- New Production Experiment Manager Models ---

class EvaluationResultEntry(BaseModel):
    __test__ = False
    test_case_id: str = Field(..., description="Unique test case ID")
    prompt: str = Field(..., description="Prompt sent to the model")
    generated_output: str = Field(..., description="Generated text completion")
    expected_output: Optional[str] = Field(None, description="Expected response text, if any")
    scores: Dict[str, float] = Field(default_factory=dict, description="Metric scores computed (e.g. Exact Match)")
    latency_ms: float = Field(..., description="Latency of the step in milliseconds")
    timestamp: datetime = Field(..., description="UTC completion timestamp")
    passed: bool = Field(..., description="Passed status of all metrics")
    success: bool = Field(True, description="Indicates if the test case generated and evaluated successfully")
    error_message: Optional[str] = Field(None, description="Error message if this specific step failed")


class Experiment(BaseModel):
    __test__ = False
    experiment_id: str = Field(..., description="Unique experiment identifier")
    timestamp: datetime = Field(..., description="UTC timestamp of the experiment completion")
    dataset_name: str = Field(..., description="Name of the dataset evaluated")
    dataset_version: Optional[str] = Field(None, description="Version identifier of the dataset")
    provider: str = Field(..., description="Name of the LLM provider, e.g., gemini")
    model: str = Field(..., description="The model ID used, e.g., gemini-2.5-flash")
    evaluation_configuration: Dict[str, Any] = Field(default_factory=dict, description="Evaluation generation config details")
    evaluation_results: List[EvaluationResultEntry] = Field(default_factory=list, description="Array of detailed evaluation results")
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="Consolidated average scores per metric")
    average_latency: float = Field(..., description="Average latency per test case in milliseconds")
    total_number_of_test_cases: int = Field(..., description="Total test cases evaluated")
    passed_test_cases: int = Field(..., description="Number of test cases that passed all metrics")
    failed_test_cases: int = Field(..., description="Number of test cases failing at least one metric")
