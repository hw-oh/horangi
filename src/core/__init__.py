"""
핵심 로직 모듈

- factory: Task 생성
- loaders: 데이터 로딩
- transforms: 데이터 변환
- benchmark_config: 벤치마크 설정 스키마
- config_loader: 설정 파일 로드 및 통합
- leaderboard: Weave 리더보드 생성
- leaderboard_table: 리더보드 테이블 빌더 (GLP/ALT 점수 계산)
- create_leaderboard_cli: 리더보드 생성 CLI
"""

from core.loaders import load_weave_data, load_jsonl_data
from core.answer_format import ANSWER_FORMAT
from core.benchmark_config import BenchmarkConfig
from core.config_loader import ConfigLoader, get_config, load_config


def create_benchmark(*args, **kwargs):
    """Lazy import to avoid circular dependency"""
    from core.factory import create_benchmark as _create_benchmark
    return _create_benchmark(*args, **kwargs)


def create_leaderboard(*args, **kwargs):
    """Lazy import for leaderboard creation"""
    from core.leaderboard import create_leaderboard as _create_leaderboard
    return _create_leaderboard(*args, **kwargs)


def update_leaderboard_from_weave_traces(*args, **kwargs):
    """Lazy import for leaderboard update from Weave traces"""
    from core.leaderboard import update_leaderboard_from_weave_traces as _update
    return _update(*args, **kwargs)


def create_leaderboard_table(*args, **kwargs):
    """Lazy import for leaderboard table creation"""
    from core.leaderboard_table import LeaderboardTableBuilder
    return LeaderboardTableBuilder(*args, **kwargs)


def create_leaderboard_from_benchmarks(*args, **kwargs):
    """Lazy import for leaderboard from benchmarks"""
    from core.leaderboard_table import create_leaderboard_from_benchmarks as _create
    return _create(*args, **kwargs)


def run_leaderboard_cli():
    """Run leaderboard CLI"""
    from core.create_leaderboard_cli import create_leaderboard_cli
    return create_leaderboard_cli()


__all__ = [
    "create_benchmark",
    "create_leaderboard",
    "update_leaderboard_from_weave_traces",
    "create_leaderboard_table",
    "create_leaderboard_from_benchmarks",
    "run_leaderboard_cli",
    "load_weave_data",
    "load_jsonl_data",
    "ANSWER_FORMAT",
    "BenchmarkConfig",
    "ConfigLoader",
    "get_config",
    "load_config",
]
