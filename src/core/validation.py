"""
Pydantic models for configuration validation.

These models provide runtime validation for configuration files,
ensuring that required fields are present and values are of correct types.

Usage:
    from core.validation import validate_model_config

    # Validate a model config dict
    config = yaml.safe_load(open("configs/models/gpt-4o.yaml"))
    validated = validate_model_config(config)
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ModelParamsModel(BaseModel):
    """Model generation parameters with validation."""

    max_tokens: Optional[int] = Field(default=4096, ge=1)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    reasoning_effort: Optional[Literal["low", "high", "xhigh"]] = None
    reasoning_tokens: Optional[int] = Field(default=None, ge=1)
    timeout: Optional[int] = Field(default=None, ge=1)
    max_retries: Optional[int] = Field(default=None, ge=0)
    client_timeout: Optional[int] = Field(default=None, ge=1)
    extra_body: Optional[dict] = None

    model_config = {"extra": "allow"}


class ModelSectionModel(BaseModel):
    """Model configuration section with validation."""

    name: str = Field(..., min_length=1, description="Model name")
    client: Literal["litellm", "openai"] = Field(
        default="openai", description="API client type"
    )
    provider: Optional[str] = Field(
        default=None, description="Model provider for LiteLLM routing"
    )
    base_url: Optional[str] = Field(default=None, description="Custom API base URL")
    api_key_env: Optional[str] = Field(
        default=None, description="Environment variable name for API key"
    )
    params: Optional[ModelParamsModel] = Field(default_factory=ModelParamsModel)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v


class ModelMetadataModel(BaseModel):
    """Model metadata with validation."""

    release_date: Optional[str] = None
    size_category: Optional[Literal["Small (<10B)", "Medium (10-30B)", "Large (30B<)"]] = None
    model_size: Optional[str | int] = None
    context_window: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)

    model_config = {"extra": "allow"}


class WandbSectionModel(BaseModel):
    """W&B configuration section with validation."""

    run_name: Optional[str] = None


class BenchmarkOverridesModel(BaseModel):
    """Benchmark-specific parameter overrides with validation."""

    use_native_tools: Optional[bool] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    timeout: Optional[int] = Field(default=None, ge=1)
    client_timeout: Optional[int] = Field(default=None, ge=1)
    extra_body: Optional[dict] = None

    model_config = {"extra": "allow"}


class VLLMConfigModel(BaseModel):
    """vLLM server configuration with validation."""

    model_path: str = Field(..., min_length=1, description="HuggingFace model ID or local path")
    tensor_parallel_size: Optional[int] = Field(default=1, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    port: Optional[int] = Field(default=8000, ge=1, le=65535)
    host: Optional[str] = Field(default="0.0.0.0")
    max_model_len: Optional[int] = Field(default=None, ge=1)
    trust_remote_code: Optional[bool] = Field(default=True)
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None

    model_config = {"extra": "allow"}


class ModelConfigModel(BaseModel):
    """Complete model configuration file with validation."""

    wandb: Optional[WandbSectionModel] = None
    metadata: Optional[ModelMetadataModel] = None
    model: ModelSectionModel
    benchmarks: Optional[dict[str, BenchmarkOverridesModel]] = None
    vllm: Optional[VLLMConfigModel] = None

    model_config = {"extra": "allow"}


def validate_model_config(config: dict) -> ModelConfigModel:
    """
    Validate a model configuration dictionary.

    Args:
        config: Raw configuration dictionary loaded from YAML

    Returns:
        Validated ModelConfigModel instance

    Raises:
        pydantic.ValidationError: If validation fails
    """
    return ModelConfigModel.model_validate(config)


def validate_model_config_safe(config: dict) -> tuple[ModelConfigModel | None, str | None]:
    """
    Validate a model configuration dictionary safely.

    Args:
        config: Raw configuration dictionary loaded from YAML

    Returns:
        Tuple of (validated config or None, error message or None)
    """
    try:
        validated = ModelConfigModel.model_validate(config)
        return validated, None
    except Exception as e:
        return None, str(e)
