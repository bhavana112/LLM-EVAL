from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

class TokenUsage(BaseModel):
    input_tokens: Optional[int] = Field(None, description="Number of tokens in prompt input")
    output_tokens: Optional[int] = Field(None, description="Number of tokens generated")
    total_tokens: Optional[int] = Field(None, description="Total tokens used (input + output)")


class LLMResponse(BaseModel):
    text: str = Field(..., description="The generated output text string")
    provider_name: str = Field(..., description="Name of the LLM provider, e.g. gemini")
    model_name: str = Field(..., description="The model ID used, e.g. gemini-2.5-flash")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    timestamp: datetime = Field(..., description="UTC timestamp of request completion")
    request_id: str = Field(..., description="Unique UUID identifying the request")
    success: bool = Field(default=True, description="True if the request completed successfully, False otherwise")
    error_message: Optional[str] = Field(None, description="Error details if success is False")
    token_usage: Optional[TokenUsage] = Field(None, description="Token usage statistics if provided by model response")
    raw_response: Optional[Any] = Field(None, description="Unfiltered raw API response dictionary or object")
