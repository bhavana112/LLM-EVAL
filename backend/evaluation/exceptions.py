from typing import Optional

class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class InvalidDatasetError(EvaluationError):
    """Raised when an evaluation dataset fails schema validation or file reading."""
    pass


class MetricExecutionError(EvaluationError):
    """Raised when a DeepEval metric fails during execution on a test case."""
    pass
