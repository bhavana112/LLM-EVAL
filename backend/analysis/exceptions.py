from typing import Optional
from backend.core.exceptions import PlatformException, ValidationException

class FailureAnalysisError(PlatformException):
    """Base exception class for all failure analysis errors."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message, status_code=500)


class NoFailedCasesError(ValidationException, FailureAnalysisError):
    """Raised when there are no failed cases in the experiment to analyze."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 400


class InvalidExperimentDataError(ValidationException, FailureAnalysisError):
    """Raised when the experiment data or results are malformed or invalid."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 400


class AnalysisExecutionError(FailureAnalysisError):
    """Raised when the LLM provider fails or analysis execution fails."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.status_code = 500

