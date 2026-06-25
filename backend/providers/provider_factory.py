from typing import Dict, Type, Optional
from backend.providers.base import LLMProvider
from backend.providers.gemini_provider import GeminiProvider
from backend.providers.exceptions import InvalidModelError

class ProviderFactory:
    """Registry and factory for constructing LLM providers dynamically."""
    
    _providers: Dict[str, Type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(
        cls, 
        provider: str, 
        model: str, 
        api_key: Optional[str] = None
    ) -> LLMProvider:
        """
        Instantiate and return the requested provider.
        """
        provider_name = provider.lower()
        provider_cls = cls._providers.get(provider_name)
        
        if not provider_cls:
            raise InvalidModelError(
                f"Unsupported provider: '{provider}'. Registered: {list(cls._providers.keys())}",
                provider=provider
            )
            
        return provider_cls(model_name=model, api_key=api_key)

# Register default provider
ProviderFactory.register("gemini", GeminiProvider)
