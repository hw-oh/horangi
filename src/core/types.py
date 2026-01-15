"""
Type definitions for Horangi configuration structures.

These TypedDict definitions provide better IDE support and type checking
for configuration dictionaries used throughout the codebase.
"""

from typing import Literal, TypedDict


class FieldMapping(TypedDict, total=False):
    """Mapping from dataset fields to Sample fields."""

    id: str | list[str]
    input: str | list[str]
    target: str | list[str]
    choices: str | list[str]


class BenchmarkMetadata(TypedDict, total=False):
    """Benchmark metadata configuration."""

    supports_dynamic_solver: bool
    native_solver: str
    text_solver: str


class BenchmarkConfigDict(TypedDict, total=False):
    """Configuration for a single benchmark."""

    # Data source
    data_type: Literal["weave", "jsonl"]
    data_source: str
    split: str

    # Field mapping
    field_mapping: FieldMapping
    answer_format: str
    default_fields: dict[str, str]

    # Solver and scorer
    solver: str
    solver_args: dict
    scorer: str
    system_message: str

    # Sampling
    sampling: Literal["stratified", "balanced"] | None
    sampling_by: str

    # Base task (for inheriting from inspect_evals)
    base: str

    # Metadata
    metadata: BenchmarkMetadata


class ModelParams(TypedDict, total=False):
    """Model generation parameters."""

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    reasoning_effort: Literal["low", "high", "xhigh"]
    reasoning_tokens: int
    timeout: int
    max_retries: int
    client_timeout: int
    extra_body: dict


class ModelSection(TypedDict, total=False):
    """Model configuration section."""

    name: str
    client: Literal["litellm", "openai"]
    provider: str
    base_url: str
    api_key_env: str
    params: ModelParams


class ModelMetadata(TypedDict, total=False):
    """Model metadata."""

    release_date: str
    size_category: Literal["Small (<10B)", "Medium (10-30B)", "Large (30B<)"] | None
    model_size: str | int | None
    context_window: int
    max_output_tokens: int


class WandbSection(TypedDict, total=False):
    """W&B configuration section."""

    run_name: str


class BenchmarkOverrides(TypedDict, total=False):
    """Benchmark-specific parameter overrides."""

    use_native_tools: bool
    max_tokens: int
    temperature: float
    timeout: int
    client_timeout: int
    extra_body: dict


class VLLMConfig(TypedDict, total=False):
    """vLLM server configuration."""

    model_path: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    port: int
    host: str
    max_model_len: int
    trust_remote_code: bool
    dtype: str
    quantization: str
    enforce_eager: bool


class ModelConfigDict(TypedDict, total=False):
    """Complete model configuration file structure."""

    wandb: WandbSection
    metadata: ModelMetadata
    model: ModelSection
    benchmarks: dict[str, BenchmarkOverrides]
    vllm: VLLMConfig


class BaseConfigDict(TypedDict, total=False):
    """Base configuration file structure."""

    testmode: bool
    defaults: ModelParams
    benchmarks: dict


class BenchmarkScoreInfo(TypedDict, total=False):
    """Score information for a benchmark result."""

    score: float | None
    details: dict[str, float]
