"""
벤치마크 팩토리

config.py의 설정을 기반으로 Inspect AI Task를 생성합니다.
"""

from pathlib import Path
from typing import Any, Literal
import importlib

import weave
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, MemoryDataset, json_dataset
from inspect_ai.scorer import Scorer, choice, match
from inspect_ai.solver import Solver, multiple_choice, generate, system_message

from horangi.benchmarks.config import (
    BENCHMARKS,
    TARGET_TRANSFORMS,
    get_benchmark_config,
)

# 데이터 디렉토리
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Scorer/Solver 매핑 (config에서 문자열로 지정)
# =============================================================================

def get_scorer_by_name(scorer_name: str) -> list[Scorer]:
    """scorer 이름으로 Scorer 인스턴스 생성"""
    scorers_map = {
        "choice": lambda: [choice()],
        "match": lambda: [match()],
        "match_numeric": lambda: [match(numeric=True)],
    }
    if scorer_name in scorers_map:
        return scorers_map[scorer_name]()
    raise ValueError(f"Unknown scorer: {scorer_name}. Available: {list(scorers_map.keys())}")


def get_solver_by_name(solver_name: str) -> list[Solver]:
    """solver 이름으로 Solver 인스턴스 생성"""
    solvers_map = {
        "multiple_choice": lambda: [multiple_choice()],
        "generate": lambda: [generate()],
    }
    if solver_name in solvers_map:
        return solvers_map[solver_name]()
    raise ValueError(f"Unknown solver: {solver_name}. Available: {list(solvers_map.keys())}")


# =============================================================================
# inspect_evals 벤치마크 매핑
# =============================================================================

def get_base_components(base_name: str) -> tuple[list[Solver], list[Scorer]] | None:
    """
    inspect_evals에서 원본 벤치마크의 solver와 scorer를 가져옵니다.
    """
    if base_name == "hellaswag":
        from inspect_evals.hellaswag.hellaswag import SYSTEM_MESSAGE
        return (
            [system_message(SYSTEM_MESSAGE), multiple_choice()],
            [choice()],
        )
    
    elif base_name == "aime2025":
        from inspect_evals.aime2025.aime2025 import aime_scorer
        from inspect_evals.aime2024.aime2024 import aime2024_solver
        return (
            [aime2024_solver()],
            [aime_scorer()],
        )
    
    elif base_name == "gsm8k":
        from inspect_evals.gsm8k.gsm8k import gsm8k_solver, expression_equivalence
        return (
            [gsm8k_solver()],
            [expression_equivalence()],
        )
    
    elif base_name == "ifeval":
        from inspect_evals.ifeval.ifeval import instruction_following
        return (
            [generate()],
            [instruction_following()],
        )
    
    # base가 없으면 None 반환
    else:
        return None




# =============================================================================
# 데이터 로딩
# =============================================================================

def load_weave_data(
    ref: str,
    split: str | None = None,
) -> list[dict]:
    """Weave에서 데이터 로드"""
    # ref에서 프로젝트 추출: weave:///entity/project/object/name:version
    # → entity/project
    parts = ref.replace("weave:///", "").split("/")
    project = f"{parts[0]}/{parts[1]}"
    
    weave.init(project)
    data = weave.ref(ref).get()
    rows = data.rows if hasattr(data, "rows") else list(data)
    
    if split:
        rows = [r for r in rows if r.get("split") == split]
    
    return [dict(r) if hasattr(r, "keys") else r for r in rows]


def load_jsonl_data(path: str) -> list[dict]:
    """로컬 JSONL 파일에서 데이터 로드"""
    import json
    
    file_path = DATA_DIR / path if not Path(path).is_absolute() else Path(path)
    
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# =============================================================================
# Sample 변환
# =============================================================================

def create_ifeval_sample(record: dict[str, Any]) -> Sample:
    """IFEval 전용 Sample 생성 (원본 로직 그대로)"""
    new_kwargs = {}
    for index in range(len(record["instruction_id_list"])):
        # remove None values from kwargs
        kwargs = {k: v for k, v in record["kwargs"][index].items() if v}
        new_kwargs[index] = kwargs

    return Sample(
        id=record["key"],
        input=record["prompt"],
        metadata={
            "prompt": record["prompt"],
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": new_kwargs,
        },
    )


def create_sample(
    record: dict[str, Any],
    field_mapping: dict[str, Any],
    target_transform: str = "identity",
) -> Sample:
    """
    config의 field_mapping을 기반으로 Sample 생성
    
    field_mapping 예시:
    {
        "id": "id",                    # 단일 필드
        "id": ["id", "ind"],          # 여러 필드 중 첫 번째 있는 것
        "input": "problem",
        "target": "answer",
        "choices": "endings",          # 선택 (객관식일 때)
    }
    """
    transform = TARGET_TRANSFORMS.get(target_transform, TARGET_TRANSFORMS["identity"])
    
    def get_field(record: dict, field_spec: Any) -> Any:
        """필드 값 가져오기 (단일 또는 대체 필드 목록)"""
        if isinstance(field_spec, list):
            for f in field_spec:
                if f in record:
                    return record[f]
            return None
        return record.get(field_spec)
    
    sample_id = get_field(record, field_mapping.get("id", "id"))
    input_val = get_field(record, field_mapping["input"])
    target_val = get_field(record, field_mapping["target"])
    choices = get_field(record, field_mapping.get("choices"))
    
    # target 변환
    target = transform(target_val)
    
    # 메타데이터 (매핑에 없는 필드들)
    mapped_fields = set()
    for v in field_mapping.values():
        if isinstance(v, list):
            mapped_fields.update(v)
        else:
            mapped_fields.add(v)
    
    metadata = {k: v for k, v in record.items() if k not in mapped_fields}
    
    # 정답을 metadata에 명시적으로 추가 (trace에서 확인 가능)
    metadata["_target"] = target
    
    return Sample(
        id=str(sample_id) if sample_id is not None else None,
        input=input_val,
        target=target,
        choices=choices,
        metadata=metadata,
    )


# =============================================================================
# 벤치마크 생성
# =============================================================================

def create_benchmark(
    name: str,
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
    use_korean_prompt: bool = True,
    **kwargs,
) -> Task:
    """
    config 기반으로 벤치마크 Task 생성
    
    Args:
        name: 벤치마크 이름 (config.py의 BENCHMARKS 키)
        shuffle: 데이터 셔플 여부
        limit: 샘플 수 제한
        split: 데이터 분할 (config 값 오버라이드)
        use_korean_prompt: 한국어 시스템 메시지 사용 여부
        **kwargs: 추가 옵션
    
    Returns:
        Task: Inspect AI Task 객체
    """
    config = get_benchmark_config(name)
    
    # 데이터 로드
    data_type = config["data_type"]
    data_source = config["data_source"]
    
    if data_type == "weave":
        data_split = split or config.get("split")
        rows = load_weave_data(
            ref=data_source,
            split=data_split,
        )
    else:  # jsonl
        rows = load_jsonl_data(data_source)
    
    # Sample 변환
    base_name = config.get("base", "")
    
    # 특수 record_to_sample이 필요한 벤치마크
    if base_name == "ifeval":
        samples = [create_ifeval_sample(r) for r in rows]
    else:
        field_mapping = config.get("field_mapping", {})
        target_transform = config.get("target_transform", "identity")
        samples = [
            create_sample(r, field_mapping, target_transform)
            for r in rows
        ]
    
    # limit 적용
    if limit:
        samples = samples[:limit]
    
    dataset = MemoryDataset(samples=samples, shuffled=shuffle)
    
    # Solver & Scorer 결정
    # 우선순위: config 직접 지정 > base(inspect_evals) > 기본값(mcqa)
    base_name = config.get("base", "")
    
    # === Scorer ===
    # 1. 커스텀 scorer (scorers.py에 정의)
    custom_scorer_name = config.get("custom_scorer")
    if custom_scorer_name:
        from horangi.benchmarks import scorers as custom_scorers
        scorer = [getattr(custom_scorers, custom_scorer_name)()]
    # 2. config에서 scorer 직접 지정
    elif config.get("scorer"):
        scorer = get_scorer_by_name(config["scorer"])
    # 3. base에서 가져오기
    elif base_name:
        components = get_base_components(base_name)
        scorer = components[1] if components else get_scorer_by_name("choice")
    # 4. 기본값
    else:
        scorer = get_scorer_by_name("choice")
    
    # === Solver ===
    # 1. config에서 solver 직접 지정
    if config.get("solver"):
        base_solver = get_solver_by_name(config["solver"])
    # 2. base에서 가져오기
    elif base_name:
        components = get_base_components(base_name)
        base_solver = components[0] if components else get_solver_by_name("multiple_choice")
    # 3. 기본값
    else:
        base_solver = get_solver_by_name("multiple_choice")
    
    # 한국어 시스템 메시지 추가
    if use_korean_prompt and config.get("system_message"):
        solver = [system_message(config["system_message"])] + base_solver
    else:
        solver = base_solver
    
    # 메타데이터
    metadata = {
        "benchmark": name,
        "base": config.get("base", ""),
        "language": "ko",
        **(config.get("metadata", {})),
    }
    
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer,
        name=name,
        metadata=metadata,
    )


# =============================================================================
# Task 데코레이터 생성
# =============================================================================

def make_task_function(benchmark_name: str):
    """벤치마크 이름으로 @task 함수 동적 생성"""
    
    @task
    def benchmark_task(
        shuffle: bool = False,
        limit: int | None = None,
        split: str | None = None,
        use_korean_prompt: bool = True,
    ) -> Task:
        return create_benchmark(
            name=benchmark_name,
            shuffle=shuffle,
            limit=limit,
            split=split,
            use_korean_prompt=use_korean_prompt,
        )
    
    # 함수 이름 변경
    benchmark_task.__name__ = benchmark_name
    benchmark_task.__doc__ = f"{benchmark_name} 벤치마크"
    
    return benchmark_task

