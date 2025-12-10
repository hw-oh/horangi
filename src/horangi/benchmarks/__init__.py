"""
한국어 벤치마크 모듈

config.py의 설정을 기반으로 벤치마크를 생성합니다.

사용법:
    from horangi.benchmarks import create_benchmark
    
    task = create_benchmark("ko_hellaswag", limit=10)

새 벤치마크 추가:
    1. config.py의 BENCHMARKS에 설정 추가
    2. (필요 시) scorers.py에 커스텀 scorer 추가
    3. eval_tasks.py에 @task 함수 추가
"""

from horangi.benchmarks.config import BENCHMARKS, list_benchmarks
from horangi.benchmarks.factory import create_benchmark

__all__ = [
    # Factory
    "create_benchmark",
    # Config
    "BENCHMARKS",
    "list_benchmarks",
]
