from typing import Optional
from backend.core.exceptions import PlatformException, StorageException, ValidationException

class ReportingError(PlatformException):
    """Base exception for all reporting and regression errors."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message, status_code=500)


class ReportGenerationError(StorageException, ReportingError):
    """Raised when generating a report from experiment details fails."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 500


class IncompatibleExperimentsError(ValidationException, ReportingError):
    """Raised when comparing experiments that are not compatible (e.g. evaluating different datasets)."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 400

