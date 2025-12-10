"""
KoHellaSwag (상속 패턴)

inspect_evals.hellaswag를 기반으로 데이터 소스만 Weave로 교체합니다.
원본 hellaswag의 설정, record_to_sample 로직을 최대한 재사용합니다.

사용 시점:
- 기존 inspect_evals 벤치마크의 한국어 버전을 만들 때
- 원본과 동일한 평가 방식을 유지하면서 데이터만 교체할 때
- inspect_evals가 이미 설치된 환경에서 사용할 때

장점:
- 원본 벤치마크와 동일한 평가 로직 보장
- 코드 중복 최소화
- 원본 업데이트 시 자동 반영

단점:
- inspect_evals 의존성 필요
- 원본 구조 변경 시 호환성 문제 가능
"""

from typing import Any, Literal

import weave
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message

# ============================================================================
# 원본 hellaswag에서 재사용
# ============================================================================
from inspect_evals.hellaswag.hellaswag import (
    SYSTEM_MESSAGE as HELLASWAG_SYSTEM_MESSAGE,
    # record_to_sample as hellaswag_record_to_sample,  # 구조가 다를 수 있어 직접 구현
)

# ============================================================================
# KoHellaSwag 설정
# ============================================================================
WEAVE_REF = "weave:///wandb-korea/evaluation-job/object/KoHellaSwag:PY229AMRxLFoCLqKsaEguY4jVCuvyoMnQ5wJ1wkrXfU"

# 한국어 시스템 메시지 (원본 번역)
SYSTEM_MESSAGE_KO = """
이야기의 가장 자연스러운 다음 문장을 선택하세요.
"""


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    원본 hellaswag.record_to_sample과 동일한 로직 적용
    
    원본 코드:
        return Sample(
            id=create_stable_id(record["ctx"], record["source_id"], prefix="hellaswag"),
            input=record["ctx"],
            target=chr(ord("A") + int(record["label"])),
            choices=record["endings"],
            metadata=dict(source_id=record["source_id"]),
        )
    """
    label = record.get("label", 0)
    target = chr(ord("A") + int(label))
    
    return Sample(
        id=str(record.get("id", record.get("ind", ""))),
        input=record["ctx"],
        target=target,
        choices=record["endings"],
        metadata={
            "source_id": record.get("source_id", ""),
            "activity_label": record.get("activity_label", ""),
        },
    )


def load_dataset_from_weave(
    split: Literal["train", "validation", "test"] | None = "train",
    shuffle: bool = False,
    limit: int | None = None,
) -> Dataset:
    """
    Weave에서 KoHellaSwag 데이터셋 로드
    
    원본 hellaswag의 hf_dataset() 호출을 대체합니다.
    """
    weave.init("wandb-korea/evaluation-job")
    data = weave.ref(WEAVE_REF).get()
    
    rows = data.rows if hasattr(data, "rows") else list(data)
    
    if split:
        rows = [r for r in rows if r.get("split") == split]
    
    samples = [record_to_sample(r) for r in rows]
    
    if limit:
        samples = samples[:limit]
    
    return MemoryDataset(samples=samples, shuffled=shuffle)


@task
def ko_hellaswag_inherited(
    shuffle: bool = False,
    split: Literal["train", "validation", "test"] | None = "train",
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """
    KoHellaSwag (상속 패턴)
    
    원본 inspect_evals.hellaswag와 동일한 구조를 사용합니다:
    - solver: [system_message, multiple_choice]
    - scorer: choice()
    
    Args:
        shuffle: 데이터 셔플 여부
        split: 데이터 분할 (train/validation/test), None이면 전체
        limit: 샘플 수 제한
        use_korean_prompt: True면 한국어 프롬프트, False면 원본 영어 프롬프트
    """
    dataset = load_dataset_from_weave(split=split, shuffle=shuffle, limit=limit)
    
    sys_msg = SYSTEM_MESSAGE_KO if use_korean_prompt else HELLASWAG_SYSTEM_MESSAGE
    
    # 원본 hellaswag와 동일한 Task 구조
    return Task(
        dataset=dataset,
        solver=[system_message(sys_msg), multiple_choice()],
        scorer=choice(),
        name="ko_hellaswag_inherited",
        metadata={
            "benchmark": "ko_hellaswag",
            "pattern": "inherited",
            "base": "inspect_evals.hellaswag",
            "language": "ko",
            "split": split,
        },
    )


# 편의를 위한 alias
ko_hellaswag = ko_hellaswag_inherited
