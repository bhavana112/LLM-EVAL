"""Analysis and failure explanation package."""

from backend.analysis.failure_analyzer import FailureAnalyzer, DEFAULT_CATEGORIES
from backend.analysis.prompt_builder import PromptBuilder
from backend.analysis.models import FailureAnalysisResult, GroupedFailure
from backend.analysis.exceptions import (
    FailureAnalysisError,
    NoFailedCasesError,
    InvalidExperimentDataError,
    AnalysisExecutionError
)
from backend.analysis.utils import extract_json

__all__ = [
    "FailureAnalyzer",
    "PromptBuilder",
    "FailureAnalysisResult",
    "GroupedFailure",
    "FailureAnalysisError",
    "NoFailedCasesError",
    "InvalidExperimentDataError",
    "AnalysisExecutionError",
    "extract_json",
    "DEFAULT_CATEGORIES"
]
