"""
벤치마크 팩토리

config를 기반으로 Inspect AI Task를 생성합니다.

핵심 아이디어: 
- inspect_evals의 원본 Task를 가져와서 dataset만 override
- 원본의 solver, scorer, 기타 설정은 그대로 사용
"""

from typing import Any
import importlib

from inspect_ai import Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Scorer, choice, match, model_graded_qa
from scorers import hle_grader
from inspect_ai.solver import Solver, multiple_choice, generate, system_message

from benchmarks import get_benchmark_config
from core.loaders import load_weave_data, load_jsonl_data
from core.answer_format import ANSWER_FORMAT


# =============================================================================
# Sampling Functions
# =============================================================================

def _stratified_sample(rows: list[dict], group_by: str, limit: int) -> list[dict]:
    """
    Stratified Sampling: 원본 비율 유지
    
    Args:
        rows: 전체 데이터
        group_by: 그룹화할 필드명 (예: "category")
        limit: 총 샘플 수
    
    Returns:
        원본 비율을 유지하며 샘플링된 데이터
    """
    from collections import defaultdict
    import random
    
    random.seed(42)
    
    # 카테고리별로 그룹화
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key].append(row)
    
    # 각 카테고리 내에서 셔플
    for category_rows in groups.values():
        random.shuffle(category_rows)
    
    # 비율에 따른 샘플 수 계산
    total = len(rows)
    samples_per_category = {}
    remaining = limit
    
    for category, category_rows in groups.items():
        ratio = len(category_rows) / total
        n_samples = int(limit * ratio)
        samples_per_category[category] = max(1, n_samples)
        remaining -= samples_per_category[category]
    
    # 나머지를 큰 카테고리부터 분배
    for category in sorted(groups.keys(), key=lambda c: len(groups[c]), reverse=True):
        if remaining <= 0:
            break
        samples_per_category[category] += 1
        remaining -= 1
    
    # 샘플링
    result = []
    for category, category_rows in groups.items():
        n_samples = samples_per_category[category]
        result.extend(category_rows[:min(n_samples, len(category_rows))])
    
    random.shuffle(result)
    return result[:limit]


def _balanced_sample(rows: list[dict], group_by: str, limit: int) -> list[dict]:
    """
    Balanced Sampling: 각 카테고리에서 동일 수 샘플링
    
    Args:
        rows: 전체 데이터
        group_by: 그룹화할 필드명 (예: "category")
        limit: 총 샘플 수
    
    Returns:
        각 카테고리에서 균등하게 샘플링된 데이터
    """
    from collections import defaultdict
    import random
    
    random.seed(42)
    
    # 카테고리별로 그룹화
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key].append(row)
    
    # 각 카테고리 내에서 셔플
    for category_rows in groups.values():
        random.shuffle(category_rows)
    
    # 균등 분배: limit / 카테고리 수
    n_categories = len(groups)
    base_per_category = limit // n_categories
    extra = limit % n_categories
    
    # 샘플링
    result = []
    for i, (category, category_rows) in enumerate(sorted(groups.items())):
        n_samples = base_per_category + (1 if i < extra else 0)
        result.extend(category_rows[:min(n_samples, len(category_rows))])
    
    random.shuffle(result)
    return result[:limit]


# =============================================================================
# 원본 Task 로딩
# =============================================================================

def get_base_task(base_module: str) -> Task:
    """
    inspect_evals에서 원본 Task를 가져옵니다.
    
    Args:
        base_module: 모듈 경로 (예: "inspect_evals.ifeval.ifeval")
                     task 함수명 = 모듈명의 마지막 부분 ("ifeval")
    """
    module = importlib.import_module(base_module)
    func_name = base_module.rsplit(".", 1)[1]
    return getattr(module, func_name)()


def get_record_to_sample(base_module: str):
    """
    원본 벤치마크의 record_to_sample 함수를 가져옵니다.
    
    Args:
        base_module: 모듈 경로 (예: "inspect_evals.ifeval.ifeval")
    """
    module = importlib.import_module(base_module)
    return getattr(module, "record_to_sample", None)


# =============================================================================
# Sample 변환 (base가 없는 벤치마크용)
# =============================================================================

def create_sample(
    record: dict[str, Any],
    field_mapping: dict[str, Any],
    answer_format: str = "index_0",
) -> Sample:
    """config의 field_mapping을 기반으로 Sample 생성"""
    transform = ANSWER_FORMAT.get(answer_format, ANSWER_FORMAT["index_0"])
    
    def get_field(record: dict, field_spec: Any) -> Any:
        if isinstance(field_spec, list):
            for f in field_spec:
                if f in record:
                    return record[f]
            return None
        return record.get(field_spec)
    
    sample_id = get_field(record, field_mapping.get("id", "id"))
    input_val = get_field(record, field_mapping["input"])
    target_val = get_field(record, field_mapping.get("target"))
    choices = get_field(record, field_mapping.get("choices"))
    
    # target 변환 (text format은 choices 필요)
    if target_val is not None:
        if answer_format == "text":
            target = transform(target_val, choices)
        else:
            target = transform(target_val)
    else:
        # target이 없는 경우 빈 문자열 (refusal 태스크 등)
        target = ""
    
    # 메타데이터
    mapped_fields = set()
    for v in field_mapping.values():
        if isinstance(v, list):
            mapped_fields.update(v)
        else:
            mapped_fields.add(v)
    
    metadata = {k: v for k, v in record.items() if k not in mapped_fields}
    if target is not None:
        metadata["_target"] = target
    
    return Sample(
        id=str(sample_id) if sample_id is not None else None,
        input=input_val,
        target=target,
        choices=choices,
        metadata=metadata,
    )


# =============================================================================
# Scorer/Solver (base가 없는 벤치마크용)
# =============================================================================

def get_scorer_by_name(scorer_name: str) -> list[Scorer]:
    """
    scorer 이름으로 Scorer 인스턴스 생성
    
    1. 내장 scorer 먼저 확인 (choice, match 등)
    2. 없으면 커스텀 scorer에서 찾기 (grid_match 등)
    """
    # 내장 scorer
    from scorers import hallulens_qa_scorer, refusal_scorer
    from scorers.swebench_server_scorer import swebench_server_scorer
    
    builtin_scorers = {
        "choice": lambda: [choice()],
        "match": lambda: [match()],
        "match_numeric": lambda: [match(numeric=True)],
        "model_graded_qa": lambda: [model_graded_qa(model="openai/gpt-4o-mini")],
        "hle_grader": lambda: [hle_grader(judge_model="openai/gpt-4o-mini")],
        # HalluLens scorers
        "hallulens_qa_scorer": lambda: [hallulens_qa_scorer()],
        "refusal_scorer": lambda: [refusal_scorer()],
        # SWE-bench scorer
        "swebench_server_scorer": lambda: [swebench_server_scorer()],
    }
    if scorer_name in builtin_scorers:
        return builtin_scorers[scorer_name]()
    
    # 커스텀 scorer
    try:
        import scorers as custom_scorers
        if hasattr(custom_scorers, scorer_name):
            return [getattr(custom_scorers, scorer_name)()]
    except ImportError:
        pass
    
    raise ValueError(f"Unknown scorer: {scorer_name}. Available: {list(builtin_scorers.keys())}")


def get_solver_by_name(solver_name: str) -> list[Solver]:
    """solver 이름으로 Solver 인스턴스 생성"""
    # 내장 solver
    from solvers.swebench_patch_solver import swebench_patch_solver
    
    builtin_solvers = {
        "multiple_choice": lambda: [multiple_choice()],
        "generate": lambda: [generate()],
        # SWE-bench solver
        "swebench_patch_solver": lambda: [swebench_patch_solver()],
    }
    if solver_name in builtin_solvers:
        return builtin_solvers[solver_name]()
    
    # 커스텀 solver
    try:
        import solvers as custom_solvers
        if hasattr(custom_solvers, solver_name):
            return [getattr(custom_solvers, solver_name)()]
    except ImportError:
        pass
    
    raise ValueError(f"Unknown solver: {solver_name}. Available: {list(builtin_solvers.keys())}")


# =============================================================================
# 벤치마크 생성
# =============================================================================

def create_benchmark(
    name: str,
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
    use_korean_prompt: bool = True,
    use_native_tools: bool | None = None,
    **kwargs,
) -> Task:
    """
    config 기반으로 벤치마크 Task 생성
    
    base가 있으면: 원본 Task를 가져와서 dataset만 override
    base가 없으면: config의 solver/scorer로 새 Task 생성
    
    Args:
        name: 벤치마크 이름
        shuffle: 데이터 셔플 여부
        limit: 샘플 수 제한
        split: 데이터 분할
        use_korean_prompt: 한국어 프롬프트 사용 여부
        use_native_tools: tool calling 사용 여부 (BFCL 등에서 사용)
                         None이면 기본값 (native) 사용
    """
    config = get_benchmark_config(name)
    
    # 동적 solver 선택 (BFCL 등)
    if use_native_tools is not None and config.get("metadata", {}).get("supports_dynamic_solver"):
        if use_native_tools:
            config["solver"] = config.get("metadata", {}).get("native_solver", config.get("solver"))
        else:
            config["solver"] = config.get("metadata", {}).get("text_solver", config.get("solver"))
    base_path = config.get("base", "")
    
    # =========================================================================
    # 1. 데이터 로드
    # =========================================================================
    data_type = config["data_type"]
    data_source = config["data_source"]
    
    if data_type == "weave":
        data_split = split or config.get("split")
        rows = load_weave_data(ref=data_source, split=data_split)
    else:
        rows = load_jsonl_data(data_source)
    
    # Sampling (limit 적용 전)
    # sampling: "stratified" (비율 유지) | "balanced" (균등) | None (앞에서 자르기)
    # sampling_by: 그룹화할 필드명 (예: "category")
    sampling = config.get("sampling")
    sampling_by = config.get("sampling_by")
    
    # limit이 문자열 "None"으로 전달될 수 있으므로 처리
    if isinstance(limit, str):
        limit = None if limit.lower() == "none" else int(limit)
    
    if limit and sampling and sampling_by:
        if sampling == "stratified":
            rows = _stratified_sample(rows, sampling_by, limit)
        elif sampling == "balanced":
            rows = _balanced_sample(rows, sampling_by, limit)
        else:
            rows = rows[:limit]
    elif limit:
        rows = rows[:limit]
    
    # 누락된 필드에 기본값 추가
    default_fields = config.get("default_fields", {})
    if default_fields:
        for row in rows:
            for field, default_value in default_fields.items():
                if field not in row:
                    row[field] = default_value
    
    # =========================================================================
    # 2. base가 있으면: 원본 Task의 dataset만 override
    # =========================================================================
    if base_path:
        original_task = get_base_task(base_path)
        
        record_to_sample = get_record_to_sample(base_path)
        if record_to_sample:
            samples = [record_to_sample(r) for r in rows]
        else:
            field_mapping = config.get("field_mapping", {})
            answer_format = config.get("answer_format", "index_0")
            samples = [create_sample(r, field_mapping, answer_format) for r in rows]
        
        dataset = MemoryDataset(samples=samples, shuffled=shuffle, name=name)
        
        solver = original_task.solver
        scorer = original_task.scorer
        
        if use_korean_prompt and config.get("system_message"):
            solver = [system_message(config["system_message"])] + list(solver)
        
        return Task(
            dataset=dataset,
            solver=solver,
            scorer=scorer,
            name=name,
            metadata={
                "benchmark": name,
                "base": base_path,
                "language": "ko",
                **(config.get("metadata", {})),
            },
        )
    
    # =========================================================================
    # 3. base가 없으면: config의 solver/scorer로 새 Task 생성
    # =========================================================================
    else:
        field_mapping = config.get("field_mapping", {})
        answer_format = config.get("answer_format", "index_0")
        samples = [create_sample(r, field_mapping, answer_format) for r in rows]
        dataset = MemoryDataset(samples=samples, shuffled=shuffle, name=name)
        
        # Scorer
        if config.get("scorer"):
            scorer = get_scorer_by_name(config["scorer"])
        else:
            scorer = get_scorer_by_name("choice")
        
        # Solver
        if config.get("solver"):
            solver = get_solver_by_name(config["solver"])
        else:
            solver = get_solver_by_name("multiple_choice")
        
        if use_korean_prompt and config.get("system_message"):
            solver = [system_message(config["system_message"])] + solver
        
        return Task(
            dataset=dataset,
            solver=solver,
            scorer=scorer,
            name=name,
            metadata={
                "benchmark": name,
                "language": "ko",
                **(config.get("metadata", {})),
            },
        )
