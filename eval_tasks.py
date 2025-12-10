"""
Inspect CLI에서 직접 실행 가능한 Task 정의 파일

사용법:
    # 벤치마크 실행
    inspect eval eval_tasks.py@ko_hellaswag --model openai/gpt-4o -T limit=5
    inspect eval eval_tasks.py@ko_aime2025 --model openai/gpt-4o -T limit=3

    # 옵션
    -T shuffle=true      # 데이터 셔플
    -T limit=10          # 샘플 수 제한
    -T split=train       # 데이터 분할 (weave 타입)
    -T use_korean_prompt=false  # 영어 프롬프트 사용

새 벤치마크 추가:
    1. src/horangi/benchmarks/config.py의 BENCHMARKS에 설정 추가
    2. (필요 시) scorers.py에 커스텀 scorer 추가
    3. 이 파일에 @task 함수 추가

inspect-wandb가 설치되어 있으면 자동으로 WandB/Weave에 로깅됩니다.
"""

import sys
from pathlib import Path

# 프로젝트 소스를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_ai import Task, task
from horangi.benchmarks.factory import create_benchmark

# =============================================================================
# 벤치마크 Task 정의
# =============================================================================

@task
def ko_hellaswag(
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoHellaSwag"""
    return create_benchmark(
        name="ko_hellaswag",
        shuffle=shuffle,
        limit=limit,
        split=split,
        use_korean_prompt=use_korean_prompt,
    )


@task
def ko_aime2025(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoAIME2025"""
    return create_benchmark(
        name="ko_aime2025",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
    )


@task
def ko_balt_700(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoBALT-700"""
    return create_benchmark(
        name="ko_balt_700",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
    )


@task
def ifeval_ko(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """IFEval-Ko"""
    return create_benchmark(
        name="ifeval_ko",
        shuffle=shuffle,
        limit=limit,
    )
