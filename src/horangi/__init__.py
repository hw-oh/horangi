"""
Horangi: 한국어 LLM 벤치마크 평가 프레임워크

Inspect AI와 WandB/Weave를 통합하여 한국어 LLM 평가를 수행합니다.
"""

__version__ = "0.1.0"

from horangi.benchmarks import (
    ko_hellaswag,
    ko_hellaswag_inherited,
    ko_aime2025,
)

__all__ = [
    "ko_hellaswag",
    "ko_hellaswag_inherited",
    "ko_aime2025",
]
