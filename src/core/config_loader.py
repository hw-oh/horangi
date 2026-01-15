"""
Configuration Loader

Loads and integrates base_config.yaml + models/<model_name>.yaml configurations.

New Config Structure (v2):
    wandb:
      run_name: "model-name: effort"
    
    metadata:
      size_category: null
      model_size: null
      context_window: 200000
      max_output_tokens: 64000
    
    model:
      name: claude-opus-4-5-20251101
      client: litellm | openai
      provider: anthropic | openai | xai | qwen | lgai | upstage | google
      base_url: null  # for OpenAI-compatible APIs
      api_key_env: ANTHROPIC_API_KEY
      
      params:
        max_tokens: 64000
        temperature: 1.0
        reasoning_effort: high
        timeout: 7200
        max_retries: 10
    
    benchmarks:
      bfcl:
        use_native_tools: true
        max_tokens: 32000

Usage:
    from core.config_loader import ConfigLoader
    
    config = ConfigLoader()
    
    # Load full configuration
    full_config = config.load("gpt-4o")
    
    # Individual access
    model_config = config.get_model("gpt-4o")
    defaults = config.defaults
    
    # W&B settings use environment variables:
    #   WANDB_ENTITY, WANDB_PROJECT
"""

import os
from pathlib import Path
from typing import Any, Optional
import yaml

from core.types import (
    BaseConfigDict,
    ModelConfigDict,
    ModelParams,
    ModelMetadata,
    BenchmarkOverrides,
    VLLMConfig,
)

# Flag to enable strict config validation
ENABLE_STRICT_VALIDATION = os.environ.get("HORANGI_STRICT_VALIDATION", "").lower() == "true"


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary with override values taking precedence.
        Nested dicts are recursively merged.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigLoader:
    """Configuration loader class"""
    
    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Args:
            config_dir: Path to configs directory. Auto-detected if None
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Find configs directory at project root
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "configs").exists():
                    self.config_dir = current / "configs"
                    break
                current = current.parent
            else:
                raise FileNotFoundError("configs directory not found")
        
        self._base_config: Optional[dict] = None
        self._model_configs: dict[str, dict] = {}
    
    @property
    def base_config_path(self) -> Path:
        return self.config_dir / "base_config.yaml"
    
    @property
    def models_dir(self) -> Path:
        return self.config_dir / "models"
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML file"""
        if not path.exists():
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    @property
    def base(self) -> dict:
        """Load base_config.yaml (cached)"""
        if self._base_config is None:
            self._base_config = self._load_yaml(self.base_config_path)
        return self._base_config
    
    @property
    def wandb(self) -> dict:
        """WandB settings"""
        return self.base.get("wandb", {})
    
    @property
    def defaults(self) -> dict:
        """Default settings"""
        return self.base.get("defaults", {})
    
    @property
    def benchmarks(self) -> dict:
        """Benchmark common settings"""
        return self.base.get("benchmarks", {})
    
    @property
    def testmode(self) -> bool:
        """Test mode flag"""
        return self.base.get("testmode", False)
    
    def get_model(self, model_name: str, validate: bool | None = None) -> ModelConfigDict:
        """
        Load model configuration.

        Args:
            model_name: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
                       Specify without file extension
            validate: Whether to validate config with pydantic.
                     If None, uses HORANGI_STRICT_VALIDATION env var.

        Returns:
            Model configuration dictionary

        Raises:
            pydantic.ValidationError: If validation is enabled and config is invalid
        """
        if model_name in self._model_configs:
            return self._model_configs[model_name]

        # Normalize filename: openai/gpt-4o -> gpt-4o
        # If slash exists, use only the last part
        if "/" in model_name:
            file_name = model_name.split("/")[-1]
        else:
            file_name = model_name

        model_path = self.models_dir / f"{file_name}.yaml"
        model_config: ModelConfigDict = self._load_yaml(model_path)

        # Optionally validate config
        should_validate = validate if validate is not None else ENABLE_STRICT_VALIDATION
        if should_validate and model_config:
            from core.validation import validate_model_config
            validate_model_config(model_config)

        # Cache
        self._model_configs[model_name] = model_config

        return model_config
    
    def list_models(self) -> list[str]:
        """List available model configurations"""
        if not self.models_dir.exists():
            return []
        
        return [
            p.stem
            for p in self.models_dir.glob("*.yaml")
            if p.is_file()
        ]
    
    def load(self, model_name: Optional[str] = None) -> dict:
        """
        Load full configuration (base + model merged)
        
        Args:
            model_name: Model name. Returns base_config only if None
        
        Returns:
            Merged configuration dictionary
            {
                "wandb": {...},
                "defaults": {...},
                "benchmarks": {...},
                "testmode": bool,
                "model": {...}  # If model_name is specified
            }
        """
        config = self.base.copy()
        
        if model_name:
            model_config = self.get_model(model_name)
            config["model"] = model_config
            
            # Merge model.params with base defaults (for new structure)
            model_params = model_config.get("model", {}).get("params", {})
            if model_params:
                config["defaults"] = deep_merge(
                    config.get("defaults", {}),
                    model_params
                )
        
        return config
    
    # =========================================================================
    # New Config Structure Helpers (v2)
    # =========================================================================
    
    def get_model_name(self, config_name: str) -> str:
        """
        Get model name from config
        
        New structure: model.name
        
        Returns:
            Model name (e.g., "claude-opus-4-5-20251101", "gpt-5.2-2025-12-11")
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        return model_section.get("name", config_name)
    
    def get_model_client(self, config_name: str) -> str:
        """
        Get API client type
        
        New structure: model.client
        
        Returns:
            "litellm" or "openai"
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        return model_section.get("client", "openai")
    
    def get_model_provider(self, config_name: str) -> Optional[str]:
        """
        Get model provider (for LiteLLM routing)
        
        New structure: model.provider
        
        Returns:
            Provider name (e.g., "anthropic", "openai", "xai", "qwen")
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        return model_section.get("provider")
    
    def get_model_base_url(self, config_name: str) -> Optional[str]:
        """
        Get custom API base URL
        
        New structure: model.base_url
        If model.base_url is not set but vllm section exists,
        automatically generates URL from vllm.host and vllm.port.
        
        Returns:
            Base URL for OpenAI-compatible APIs, or None
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        
        # First, check explicit base_url in model section
        base_url = model_section.get("base_url")
        if base_url:
            return base_url
        
        # If no explicit base_url, check if vllm section exists
        # and auto-generate base_url from vllm.host and vllm.port
        vllm_section = model_config.get("vllm")
        if vllm_section and vllm_section.get("model_path"):
            host = vllm_section.get("host", "localhost")
            port = vllm_section.get("port", 8000)
            # Use localhost for connection even if host is 0.0.0.0
            if host == "0.0.0.0":
                host = "localhost"
            return f"http://{host}:{port}/v1"
        
        return None
    
    def get_model_api_key_env(self, config_name: str) -> Optional[str]:
        """
        Get API key environment variable name
        
        New structure: model.api_key_env
        
        Returns:
            Environment variable name (e.g., "ANTHROPIC_API_KEY")
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        return model_section.get("api_key_env")
    
    def get_model_api_key(self, config_name: str) -> Optional[str]:
        """
        Get API key value from environment
        
        Returns:
            API key value
        """
        env_name = self.get_model_api_key_env(config_name)
        if env_name:
            return os.environ.get(env_name)
        return None
    
    def get_model_params(self, config_name: str) -> ModelParams:
        """
        Get model generation parameters.

        New structure: model.params

        Returns:
            Parameters dict (temperature, max_tokens, etc.)
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        return model_section.get("params", {})
    
    def get_wandb_run_name(self, config_name: str) -> Optional[str]:
        """
        Get W&B run name
        
        New structure: wandb.run_name
        
        Returns:
            Run name for W&B, or None
        """
        model_config = self.get_model(config_name)
        wandb_section = model_config.get("wandb", {})
        return wandb_section.get("run_name")
    
    def get_metadata(self, config_name: str) -> ModelMetadata:
        """
        Get model metadata.

        New structure: metadata

        Returns:
            Metadata dict (context_window, max_output_tokens, etc.)
        """
        model_config = self.get_model(config_name)
        return model_config.get("metadata", {})
    
    def get_benchmark_config(self, config_name: str, benchmark: str) -> BenchmarkOverrides:
        """
        Get benchmark-specific configuration.

        New structure: benchmarks.<benchmark_name>

        Returns:
            Benchmark config dict (use_native_tools, max_tokens, etc.)
        """
        model_config = self.get_model(config_name)
        return model_config.get("benchmarks", {}).get(benchmark, {})
    
    def get_inspect_model_string(self, config_name: str) -> str:
        """
        Build Inspect AI model string
        
        Logic:
            - client=litellm: "litellm/{provider}/{name}"
            - client=openai: "openai/{name}"
        
        Returns:
            Model string for Inspect AI (e.g., "litellm/anthropic/claude-opus-4-5-20251101")
        """
        client = self.get_model_client(config_name)
        name = self.get_model_name(config_name)
        provider = self.get_model_provider(config_name)
        
        if client == "litellm":
            if provider:
                return f"litellm/{provider}/{name}"
            else:
                return f"litellm/{name}"
        else:  # openai
            return f"openai/{name}"
    
    def get_inspect_model_args(self, config_name: str, benchmark: Optional[str] = None) -> dict:
        """
        Return arguments needed for Inspect AI model creation
        
        Returns:
            {
                "model": "litellm/anthropic/claude-opus-4-5-20251101",
                "api_key": "...",  # if needed
                "client_timeout": 600,  # if specified
            }
        """
        model_config = self.get_model(config_name)
        model_section = model_config.get("model", {})
        
        args = {}
        
        # API key (only for non-standard providers)
        api_key_env = model_section.get("api_key_env")
        if api_key_env and api_key_env != "OPENAI_API_KEY":
            api_key = os.environ.get(api_key_env)
            if api_key:
                args["api_key"] = api_key
        
        # Client timeout and request timeout
        client = model_section.get("client", "openai")
        params = model_section.get("params", {})
        benchmark_overrides = model_config.get("benchmarks", {}).get(benchmark, {}) if benchmark else {}
        
        # client_timeout for OpenAI provider
        client_timeout = benchmark_overrides.get("client_timeout", params.get("client_timeout"))
        if client_timeout is not None:
            if client == "openai":
                args["client_timeout"] = float(client_timeout)
        
        # timeout for litellm provider (passed to acompletion)
        timeout = benchmark_overrides.get("timeout", params.get("timeout"))
        if timeout is not None:
            if client == "litellm":
                args["timeout"] = float(timeout)
        
        return args
    
    # =========================================================================
    # vLLM Server Configuration
    # =========================================================================
    
    def get_vllm_config(self, config_name: str) -> VLLMConfig | None:
        """
        Get vLLM server configuration.

        New structure: vllm section (top-level)

        Returns:
            vLLM config dict if present, None otherwise
        """
        model_config = self.get_model(config_name)
        vllm_section = model_config.get("vllm")

        if not vllm_section:
            return None

        return vllm_section
    
    def has_vllm_config(self, config_name: str) -> bool:
        """
        Check if vLLM server configuration exists
        
        Returns:
            True if vllm section is present and has model_path
        """
        vllm_config = self.get_vllm_config(config_name)
        return vllm_config is not None and "model_path" in vllm_config
    
    def get_vllm_base_url(self, config_name: str) -> Optional[str]:
        """
        Get vLLM server base URL (constructed from port)
        
        Returns:
            Base URL (e.g., "http://localhost:8000/v1") or None
        """
        vllm_config = self.get_vllm_config(config_name)
        if not vllm_config:
            return None
        
        port = vllm_config.get("port", 8000)
        return f"http://localhost:{port}/v1"


# Global instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Return global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(model_name: Optional[str] = None) -> dict:
    """Load configuration (convenience function)"""
    return get_config().load(model_name)
