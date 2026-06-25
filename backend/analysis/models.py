from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class GroupedFailure(BaseModel):
    __test__ = False
    test_case_ids: List[str] = Field(..., description="List of test case IDs grouped under this failure theme")
    description: str = Field(..., description="Theme description explaining what went wrong")


class FailureAnalysisResult(BaseModel):
    __test__ = False
    experiment_id: str = Field(..., description="The ID of the analyzed experiment")
    analysis_timestamp: datetime = Field(..., description="Timestamp of when the failure analysis ran")
    number_of_failed_cases: int = Field(..., description="Number of failed test cases analyzed")
    detected_categories: List[str] = Field(..., description="List of unique failure categories detected")
    grouped_failures: Dict[str, List[GroupedFailure]] = Field(
        ..., 
        description="Failures grouped under categorical themes mapping to themes containing list of case IDs and descriptions"
    )
    category_counts: Dict[str, int] = Field(..., description="Count of failed cases in each category")
    identified_patterns: List[str] = Field(..., description="List of recurring pattern explanations identified by the AI")
    ai_generated_summary: str = Field(..., description="AI generated high-level summary of the failures")
    recommendations: List[str] = Field(..., description="Practical recommendations for improvements")
    provider_used_for_analysis: str = Field(..., description="Name of LLM provider used for analysis")
