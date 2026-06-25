from typing import Optional
from backend.core.exceptions import ProviderException

class ProviderError(ProviderException):
    """Base exception for all provider-related errors."""
    def __init__(
        self, 
        message: str, 
        provider: Optional[str] = None, 
        model: Optional[str] = None, 
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.provider = provider
        self.model = model
        self.original_exception = original_exception
        super().__init__(self.message, status_code=502)



class AuthenticationError(ProviderError):
    """Raised when authentication credentials (e.g. API keys) are invalid or missing."""
    pass


class RateLimitError(ProviderError):
    """Raised when the provider's rate limits are exceeded."""
    pass


class TimeoutError(ProviderError):
    """Raised when the request to the provider times out."""
    pass


class InvalidModelError(ProviderError):
    """Raised when the requested model name is invalid or not supported by the provider."""
    pass


class ProviderUnavailableError(ProviderError):
    """Raised when the provider service is down or temporarily unreachable."""
    pass


class ConfigurationError(ProviderError):
    """Raised when the provider configuration is invalid or incomplete."""
    pass
