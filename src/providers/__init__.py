"""
Custom model providers for inspect_ai.
"""

# Import to register the provider
from .litellm_provider import litellm_api, LiteLLMAPI

# vLLM provider
from .vllm_provider import (
    VLLMConfig,
    VLLMServerManager,
    start_vllm_server,
    shutdown_vllm_server,
    get_vllm_server,
    is_vllm_client,
    get_vllm_base_url,
)

__all__ = [
    # LiteLLM
    "litellm_api",
    "LiteLLMAPI",
    # vLLM
    "VLLMConfig",
    "VLLMServerManager",
    "start_vllm_server",
    "shutdown_vllm_server",
    "get_vllm_server",
    "is_vllm_client",
    "get_vllm_base_url",
]

