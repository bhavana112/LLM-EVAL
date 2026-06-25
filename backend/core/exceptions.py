class PlatformException(Exception):
    """Base exception for all LLM platform exceptions."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class ProviderException(PlatformException):
    """Raised when an LLM provider errors."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message, status_code)

class StorageException(PlatformException):
    """Raised when storage operations fail."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)

class ValidationException(PlatformException):
    """Raised when parameters or datasets validation fails."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)
