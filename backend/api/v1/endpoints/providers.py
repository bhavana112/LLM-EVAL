from fastapi import APIRouter
from backend.providers import ProviderFactory

router = APIRouter()

@router.get("/")
async def list_providers():
    """
    List all supported LLM providers and models configured.
    """
    registered = list(ProviderFactory._providers.keys())
    provider_metadata = []
    
    if "openai" in registered:
        provider_metadata.append({
            "name": "openai",
            "displayName": "OpenAI",
            "models": ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]
        })
    if "anthropic" in registered:
        provider_metadata.append({
            "name": "anthropic",
            "displayName": "Anthropic",
            "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        })
        
    return {
        "providers": provider_metadata
    }
