from typing import Optional
from backend.core.exceptions import PlatformException, StorageException, ValidationException

class ExperimentManagerError(PlatformException):
    """Base exception for all experiment management errors."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message, status_code=500)


class ExperimentNotFoundError(ExperimentManagerError):
    """Raised when an experiment ID cannot be found in storage."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.status_code = 404


class StorageWriteError(StorageException, ExperimentManagerError):
    """Raised when writing to storage fails."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 500


class InvalidExperimentError(ValidationException, ExperimentManagerError):
    """Raised when experiment JSON is corrupted or invalid."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details
        self.status_code = 400

