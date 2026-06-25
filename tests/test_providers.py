import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from backend.providers import (
    ProviderFactory, LLMProvider, LLMResponse, TokenUsage,
    AuthenticationError, RateLimitError, TimeoutError,
    ProviderUnavailableError, ConfigurationError, InvalidModelError
)

# Define a test provider class to verify base LLMProvider features
class DummyProvider(LLMProvider):
    def __init__(self, model_name: str, api_key: str = "mock-key", generate_mock=None):
        super().__init__(model_name, api_key)
        self.generate_mock = generate_mock or AsyncMock()

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def _generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0, max_tokens: int = 1024):
        return await self.generate_mock(prompt, system_prompt, temperature, max_tokens)


@pytest.mark.asyncio
async def test_successful_generation():
    mock_func = AsyncMock(return_value=(
        "Hello response",
        {"raw": "dict"},
        TokenUsage(input_tokens=5, output_tokens=10, total_tokens=15)
    ))
    provider = DummyProvider("dummy-model", generate_mock=mock_func)
    
    response = await provider.generate(prompt="Hello", system_prompt="Be helpful")
    
    assert isinstance(response, LLMResponse)
    assert response.text == "Hello response"
    assert response.success is True
    assert response.latency_ms > 0
    assert response.token_usage.total_tokens == 15
    assert response.request_id is not None
    assert isinstance(response.timestamp, datetime)
    
    mock_func.assert_called_once_with("Hello", "Be helpful", 0.0, 1024)


@pytest.mark.asyncio
async def test_retry_on_transient_error():
    attempts = 0
    async def mock_call(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimitError("Rate limit exceeded")
        return "Succeeded", {}, TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    provider = DummyProvider("dummy-model", generate_mock=AsyncMock(side_effect=mock_call))
    
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        response = await provider.generate(prompt="Retry test")
        
        assert response.text == "Succeeded"
        assert attempts == 3
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_fatal_error():
    mock_func = AsyncMock(side_effect=AuthenticationError("Invalid API Key"))
    provider = DummyProvider("dummy-model", generate_mock=mock_func)
    
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(AuthenticationError):
            await provider.generate(prompt="Fatal test")
            
        assert mock_sleep.call_count == 0
        mock_func.assert_called_once()


@pytest.mark.asyncio
async def test_retries_exhausted():
    mock_func = AsyncMock(side_effect=TimeoutError("Request timed out"))
    provider = DummyProvider("dummy-model", generate_mock=mock_func)
    
    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(TimeoutError):
            await provider.generate(prompt="Exhaust test")
            
        assert mock_sleep.call_count == 3


def test_factory_creation():
    ProviderFactory.register("dummy", DummyProvider)
    
    provider = ProviderFactory.create("dummy", "test-model-123", api_key="test-key")
    assert isinstance(provider, DummyProvider)
    assert provider.model_name == "test-model-123"
    assert provider.api_key == "test-key"
    
    with pytest.raises(InvalidModelError):
        ProviderFactory.create("unsupported-provider-xyz", "some-model")


def test_gemini_missing_api_key():
    with patch("backend.providers.config.provider_settings.GEMINI_API_KEY", None):
        with pytest.raises(ConfigurationError) as exc_info:
            ProviderFactory.create("gemini", "gemini-2.5-flash", api_key=None)
        assert "API key is missing" in str(exc_info.value)
