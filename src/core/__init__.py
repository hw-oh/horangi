"""
핵심 로직 모듈

- factory: Task 생성
- loaders: 데이터 로딩
- transforms: 데이터 변환
- benchmark_config: 벤치마크 설정 스키마
- config_loader: 설정 파일 로드 및 통합
- weave_leaderboard: Weave UI 리더보드 자동 생성
"""

from core.loaders import load_weave_data, load_jsonl_data
from core.answer_format import ANSWER_FORMAT
from core.benchmark_config import BenchmarkConfig
from core.config_loader import ConfigLoader, get_config, load_config, deep_merge
from core.types import (
    BenchmarkConfigDict,
    ModelConfigDict,
    ModelParams,
    ModelMetadata,
    BenchmarkOverrides,
    VLLMConfig,
    BenchmarkScoreInfo,
)
from core.logging import get_logger, configure_logging
from core.validation import validate_model_config, validate_model_config_safe


def create_benchmark(*args, **kwargs):
    """Lazy import to avoid circular dependency"""
    from core.factory import create_benchmark as _create_benchmark
    return _create_benchmark(*args, **kwargs)


def create_weave_leaderboard(*args, **kwargs):
    """Lazy import for Weave UI leaderboard creation"""
    from core.weave_leaderboard import create_weave_leaderboard as _create
    return _create(*args, **kwargs)


__all__ = [
    "create_benchmark",
    "create_weave_leaderboard",
    "load_weave_data",
    "load_jsonl_data",
    "ANSWER_FORMAT",
    "BenchmarkConfig",
    "ConfigLoader",
    "get_config",
    "load_config",
    "deep_merge",
    # Type definitions
    "BenchmarkConfigDict",
    "ModelConfigDict",
    "ModelParams",
    "ModelMetadata",
    "BenchmarkOverrides",
    "VLLMConfig",
    "BenchmarkScoreInfo",
    # Logging
    "get_logger",
    "configure_logging",
    # Validation
    "validate_model_config",
    "validate_model_config_safe",
]
