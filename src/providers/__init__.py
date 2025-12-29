"""
Custom model providers for inspect_ai.
"""

# Import to register the provider
from .litellm_provider import litellm_api, LiteLLMAPI

__all__ = ["litellm_api", "LiteLLMAPI"]

