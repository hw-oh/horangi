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
from core.config_loader import ConfigLoader, get_config, load_config


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
]
