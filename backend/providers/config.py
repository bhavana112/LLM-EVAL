from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class ProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys & Endpoints
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Default Models
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    DEFAULT_OPENAI_MODEL: str = "gpt-4o"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"

    # Global Call Parameters
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 3
    LLM_DEFAULT_TEMPERATURE: float = 0.0

provider_settings = ProviderSettings()
