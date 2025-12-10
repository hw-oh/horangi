"""
Horangi: 한국어 LLM 벤치마크 평가 프레임워크

Inspect AI와 WandB/Weave를 통합하여 한국어 LLM 평가를 수행합니다.

사용법:
    from horangi import create_benchmark
    from inspect_ai import eval
    
    # 벤치마크 생성 및 실행
    task = create_benchmark("ko_hellaswag", limit=10)
    results = eval(task, model="openai/gpt-4o")
"""

__version__ = "0.1.0"

from horangi.benchmarks import (
    create_benchmark,
    list_benchmarks,
    BENCHMARKS,
)

__all__ = [
    "create_benchmark",
    "list_benchmarks",
    "BENCHMARKS",
]
