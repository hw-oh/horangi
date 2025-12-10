"""
한국어 벤치마크 모듈

각 벤치마크는 Inspect AI Task로 구현되어 있습니다.
inspect_evals 벤치마크를 기반으로 데이터 소스만 교체합니다.
"""

# KoHellaSwag (Weave 데이터)
from horangi.benchmarks.ko_hellaswag import (
    ko_hellaswag,
    ko_hellaswag_inherited,
)

# KoAIME2025 (로컬 JSONL 데이터)
from horangi.benchmarks.aime2025 import ko_aime2025

__all__ = [
    # KoHellaSwag
    "ko_hellaswag",
    "ko_hellaswag_inherited",
    # KoAIME2025
    "ko_aime2025",
]
