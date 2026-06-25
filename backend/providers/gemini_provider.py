from typing import Optional, Any, Tuple
import asyncio
from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.providers.base import LLMProvider
from backend.providers.response import TokenUsage
from backend.providers.exceptions import (
    ProviderError, AuthenticationError, RateLimitError,
    TimeoutError, InvalidModelError, ProviderUnavailableError, ConfigurationError
)

class GeminiProvider(LLMProvider):
    """
    Gemini LLM Provider wrapper utilizing the official google-genai SDK.
    Handles communication with Google Gemini model endpoints and translates errors.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        # Fetch default settings if not provided
        from backend.providers.config import provider_settings
        resolved_api_key = api_key or provider_settings.GEMINI_API_KEY
        
        super().__init__(model_name, resolved_api_key)

        if not self.api_key:
            raise ConfigurationError(
                "Gemini API key is missing or invalid. Please check configuration.",
                provider=self.provider_name,
                model=self.model_name
            )

        try:
            # Initialize unified Google GenAI client
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize Gemini SDK client: {str(e)}",
                provider=self.provider_name,
                model=self.model_name,
                original_exception=e
            )

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> Tuple[str, Any, Optional[TokenUsage]]:
        """
        Executes generation content call using Google GenAI SDK.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

        try:
            # Use unified async API client.aio
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
        except Exception as e:
            raise self._map_and_wrap_exception(e)

        # Parse generated text
        generated_text = response.text or ""

        # Parse usage details
        token_usage = None
        if response.usage_metadata:
            token_usage = TokenUsage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count
            )

        # Convert SDK response models to JSON/dict for serializability
        raw_dict = {}
        try:
            raw_dict = response.model_dump() if hasattr(response, "model_dump") else str(response)
        except Exception:
            raw_dict = {"response_string": str(response)}

        return generated_text, raw_dict, token_usage

    def _map_and_wrap_exception(self, e: Exception) -> ProviderError:
        """
        Maps official Google Gemini API exceptions to standard custom exceptions.
        """
        if isinstance(e, ProviderError):
            return e

        if isinstance(e, APIError):
            code = getattr(e, "code", None)
            message = getattr(e, "message", str(e))

            if code in (401, 403):
                return AuthenticationError(
                    f"Gemini authentication failed: {message}",
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=e
                )
            elif code == 429:
                return RateLimitError(
                    f"Gemini rate limit exceeded: {message}",
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=e
                )
            elif code == 404:
                return InvalidModelError(
                    f"Gemini model '{self.model_name}' not found: {message}",
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=e
                )
            elif code == 408:
                return TimeoutError(
                    f"Gemini request timeout: {message}",
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=e
                )
            elif code is not None and code >= 500:
                return ProviderUnavailableError(
                    f"Gemini service unavailable (status {code}): {message}",
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=e
                )

        import httpx
        if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError)):
            return TimeoutError(
                f"Gemini request timeout: {str(e)}",
                provider=self.provider_name,
                model=self.model_name,
                original_exception=e
            )
        elif isinstance(e, httpx.ConnectError):
            return ProviderUnavailableError(
                f"Gemini connection error: {str(e)}",
                provider=self.provider_name,
                model=self.model_name,
                original_exception=e
            )

        return ProviderError(
            f"Gemini execution failed: {str(e)}",
            provider=self.provider_name,
            model=self.model_name,
            original_exception=e
        )
