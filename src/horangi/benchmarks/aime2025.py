"""
KoAIME2025 (상속 패턴)

inspect_evals.aime2025를 기반으로 데이터 소스만 로컬 JSONL로 교체합니다.
원본 aime2025의 solver와 scorer를 재사용합니다.
"""

from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, MemoryDataset, json_dataset

# ============================================================================
# 원본 aime2025에서 재사용
# ============================================================================
from inspect_evals.aime2025.aime2025 import (
    USER_PROMPT_TEMPLATE as AIME_USER_PROMPT_TEMPLATE,
    aime_scorer,
)
from inspect_evals.aime2024.aime2024 import aime2024_solver

# ============================================================================
# 데이터 경로
# ============================================================================
DATA_DIR = Path(__file__).parent / "data"


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    원본 aime2025.record_to_sample과 동일한 로직
    
    데이터 구조:
    - id: 문제 ID
    - problem: 문제 내용
    - answer: 정답 (정수)
    """
    return Sample(
        id=str(record.get("id", "")),
        input=record["problem"],
        target=str(record["answer"]),
    )


@task
def ko_aime2025(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """
    KoAIME2025 (상속 패턴)
    
    원본 inspect_evals.aime2025와 동일한 구조:
    - solver: aime2024_solver (원본 재사용)
    - scorer: aime_scorer (원본 재사용 - boxed 처리 포함)
    
    Args:
        shuffle: 데이터 셔플 여부
        limit: 샘플 수 제한
    """
    # 로컬 JSONL 파일에서 데이터셋 로드
    dataset = json_dataset(
        str(DATA_DIR / "ko_aime2025.jsonl"),
        sample_fields=record_to_sample,
        shuffle=shuffle,
        limit=limit,
    )
    
    # 원본 aime2025와 동일한 Task 구조
    return Task(
        dataset=dataset,
        solver=aime2024_solver(),  # 원본 solver 재사용
        scorer=[aime_scorer()],     # 원본 scorer 재사용 (boxed 처리 포함)
        name="ko_aime2025",
        metadata={
            "benchmark": "ko_aime2025",
            "base": "inspect_evals.aime2025",
            "language": "ko",
            "task_type": "math_reasoning",
        },
    )
