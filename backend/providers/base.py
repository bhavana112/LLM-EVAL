import abc
import asyncio
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Tuple

from backend.providers.response import LLMResponse, TokenUsage
from backend.providers.exceptions import (
    ProviderError, RateLimitError, TimeoutError, ProviderUnavailableError
)
from backend.providers.config import provider_settings

logger = logging.getLogger("llm_platform.providers")

# Only retry transient errors
RETRYABLE_EXCEPTIONS = (RateLimitError, TimeoutError, ProviderUnavailableError)

class LLMProvider(abc.ABC):
    """Abstract Base Class for all LLM providers (e.g. Gemini, OpenAI, Anthropic)."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'gemini', 'openai')."""
        pass

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """
        Public entrypoint to generate completion response for a prompt.
        Handles latency measurement, exponential retry logic, and error handling.
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        retries = 0
        max_retries = provider_settings.LLM_MAX_RETRIES
        backoff = 1.0  # Initial sleep time in seconds
        
        logger.info(
            f"Starting generation request: provider={self.provider_name}, model={self.model_name}, "
            f"request_id={request_id}, temperature={temperature}"
        )

        while True:
            try:
                # Delegate to provider implementation
                # Returns a tuple of (generated_text, raw_response, token_usage)
                text, raw_response, token_usage = await self._generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                completion_time = datetime.now(timezone.utc)
                
                logger.info(
                    f"Generation successful: provider={self.provider_name}, model={self.model_name}, "
                    f"request_id={request_id}, latency_ms={latency_ms:.2f}, retries={retries}"
                )
                
                return LLMResponse(
                    text=text,
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    latency_ms=latency_ms,
                    timestamp=completion_time,
                    request_id=request_id,
                    success=True,
                    token_usage=token_usage,
                    raw_response=raw_response
                )

            except Exception as e:
                # Map standard library errors to custom exceptions
                mapped_error = self._map_and_wrap_exception(e)
                
                # Check if error is transient and retryable
                is_retryable = isinstance(mapped_error, RETRYABLE_EXCEPTIONS)
                
                if is_retryable and retries < max_retries:
                    retries += 1
                    sleep_time = backoff * (1 + 0.1 * random.random())
                    logger.warning(
                        f"Temporary failure: {type(mapped_error).__name__} ({str(mapped_error)}). "
                        f"Retrying attempt {retries}/{max_retries} in {sleep_time:.2f}s..."
                    )
                    await asyncio.sleep(sleep_time)
                    backoff *= 2.0  # double the backoff for next retry
                    continue
                
                # Non-retryable error or retries exhausted
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    f"Generation failed: provider={self.provider_name}, model={self.model_name}, "
                    f"request_id={request_id}, latency_ms={latency_ms:.2f}, retries={retries}, error={str(mapped_error)}"
                )
                
                # Raise the mapped exception to the caller
                raise mapped_error

    @abc.abstractmethod
    async def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> Tuple[str, Any, Optional[TokenUsage]]:
        """
        Internal abstract method to be implemented by providers.
        Returns a tuple of (generated_text, raw_api_response, token_usage_model).
        """
        pass

    def _map_and_wrap_exception(self, e: Exception) -> ProviderError:
        """
        Hook for subclasses to map provider-specific exceptions to custom exceptions.
        Can be overridden by subclasses. If already a ProviderError, returns it directly.
        """
        if isinstance(e, ProviderError):
            return e
            
        # Basic mappings for common Python errors
        if isinstance(e, asyncio.TimeoutError):
            return TimeoutError("Request timed out", provider=self.provider_name, model=self.model_name, original_exception=e)
            
        return ProviderError(f"Unexpected error: {str(e)}", provider=self.provider_name, model=self.model_name, original_exception=e)
