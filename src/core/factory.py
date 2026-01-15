"""
Benchmark Factory

Creates Inspect AI Tasks based on config.

Core idea: 
- Get original Task from inspect_evals and override only the dataset
- Keep original solver, scorer, and other settings as-is
"""

from typing import Any
import importlib

from inspect_ai import Task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.scorer import Scorer, choice, match, model_graded_qa
from scorers import hle_grader, math_grader
from inspect_ai.solver import Solver, multiple_choice, generate, system_message, prompt_template

from benchmarks import get_benchmark_config
from core.loaders import load_weave_data, load_jsonl_data
from core.answer_format import ANSWER_FORMAT


# =============================================================================
# Sampling Functions
# =============================================================================

def _stratified_sample(rows: list[dict], group_by: str, limit: int) -> list[dict]:
    """
    Stratified Sampling: Maintain original proportions
    
    Args:
        rows: Full data
        group_by: Field name for grouping (e.g., "category")
        limit: Total sample count
    
    Returns:
        Data sampled while maintaining original proportions
    """
    from collections import defaultdict
    import random
    
    random.seed(42)
    
    # Group by category
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key].append(row)
    
    # Shuffle within each category
    for category_rows in groups.values():
        random.shuffle(category_rows)
    
    # Calculate sample count by proportion
    total = len(rows)
    samples_per_category = {}
    remaining = limit
    
    for category, category_rows in groups.items():
        ratio = len(category_rows) / total
        n_samples = int(limit * ratio)
        samples_per_category[category] = max(1, n_samples)
        remaining -= samples_per_category[category]
    
    # Distribute remaining to larger categories first
    for category in sorted(groups.keys(), key=lambda c: len(groups[c]), reverse=True):
        if remaining <= 0:
            break
        samples_per_category[category] += 1
        remaining -= 1
    
    # Sample
    result = []
    for category, category_rows in groups.items():
        n_samples = samples_per_category[category]
        result.extend(category_rows[:min(n_samples, len(category_rows))])
    
    random.shuffle(result)
    return result[:limit]


def _balanced_sample(rows: list[dict], group_by: str, limit: int) -> list[dict]:
    """
    Balanced Sampling: Sample equal counts from each category
    
    Args:
        rows: Full data
        group_by: Field name for grouping (e.g., "category")
        limit: Total sample count
    
    Returns:
        Data with equal samples from each category
    """
    from collections import defaultdict
    import random
    
    random.seed(42)
    
    # Group by category
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(group_by, "unknown")
        groups[key].append(row)
    
    # Shuffle within each category
    for category_rows in groups.values():
        random.shuffle(category_rows)
    
    # Equal distribution: limit / number of categories
    n_categories = len(groups)
    base_per_category = limit // n_categories
    extra = limit % n_categories
    
    # Sample
    result = []
    for i, (category, category_rows) in enumerate(sorted(groups.items())):
        n_samples = base_per_category + (1 if i < extra else 0)
        result.extend(category_rows[:min(n_samples, len(category_rows))])
    
    random.shuffle(result)
    return result[:limit]


# =============================================================================
# Original Task Loading
# =============================================================================

def get_base_task(base_module: str) -> Task:
    """
    Get original Task from inspect_evals.
    
    Args:
        base_module: Module path (e.g., "inspect_evals.ifeval.ifeval")
                     task function name = last part of module name ("ifeval")
    """
    module = importlib.import_module(base_module)
    func_name = base_module.rsplit(".", 1)[1]
    return getattr(module, func_name)()


def get_record_to_sample(base_module: str):
    """
    Get record_to_sample function from original benchmark.
    
    Args:
        base_module: Module path (e.g., "inspect_evals.ifeval.ifeval")
    """
    module = importlib.import_module(base_module)
    return getattr(module, "record_to_sample", None)


# =============================================================================
# Sample Conversion (for benchmarks without base)
# =============================================================================

def create_sample(
    record: dict[str, Any],
    field_mapping: dict[str, Any],
    answer_format: str = "index_0",
) -> Sample:
    """Create Sample based on config's field_mapping"""
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
    
    # Convert target (text format needs choices)
    if target_val is not None:
        if answer_format == "text":
            target = transform(target_val, choices)
        else:
            target = transform(target_val)
    else:
        # No target = empty string (for refusal tasks, etc.)
        target = ""
    
    # Metadata
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
# Scorer/Solver (for benchmarks without base)
# =============================================================================

def get_scorer_by_name(scorer_name: str, judge_model: str | None = None) -> list[Scorer]:
    """
    Create Scorer instance from scorer name

    Args:
        scorer_name: Name of the scorer
        judge_model: Model to use for judge-based scorers (e.g., "openai/gpt-4o-mini")

    Returns:
        List of Scorer instances

    Notes:
        1. Check built-in scorers first (choice, match, etc.)
        2. Then look in custom scorers (grid_match, etc.)
    """
    from core.config_loader import get_config

    # Built-in scorers
    from scorers import hallulens_qa_scorer, refusal_scorer
    from scorers.swebench_server_scorer import swebench_server_scorer
    from inspect_evals.squad.squad import f1 as squad_f1, exact as squad_exact

    # Get judge model from config if not provided
    if judge_model is None:
        config = get_config()
        judge_model = config.benchmarks.get("judge_model", "openai/gpt-4o-mini")

    builtin_scorers = {
        "choice": lambda: [choice()],
        "match": lambda: [match()],
        "match_numeric": lambda: [match(numeric=True)],
        "model_graded_qa": lambda: [model_graded_qa(model=judge_model)],
        "hle_grader": lambda: [hle_grader(judge_model=judge_model)],
        # HalluLens scorers
        "hallulens_qa_scorer": lambda: [hallulens_qa_scorer()],
        "refusal_scorer": lambda: [refusal_scorer()],
        # SWE-bench scorer
        "swebench_server_scorer": lambda: [swebench_server_scorer()],
        # Math scorer
        "math_grader": lambda: [math_grader()],
        # SQuAD scorers (F1 + Exact Match)
        "squad": lambda: [squad_f1(), squad_exact()],
    }
    if scorer_name in builtin_scorers:
        return builtin_scorers[scorer_name]()

    # Custom scorers
    try:
        import scorers as custom_scorers
        if hasattr(custom_scorers, scorer_name):
            return [getattr(custom_scorers, scorer_name)()]
    except ImportError:
        pass

    raise ValueError(f"Unknown scorer: {scorer_name}. Available: {list(builtin_scorers.keys())}")


def get_solver_by_name(solver_name: str, solver_args: dict | None = None) -> list[Solver]:
    """
    Create Solver instance from solver name
    
    Args:
        solver_name: Name of the solver
        solver_args: Optional arguments to pass to the solver
    """
    solver_args = solver_args or {}
    
    # Built-in solvers
    from solvers.swebench_patch_solver import swebench_patch_solver
    
    def _generate_with_template(**kw):
        """generate solver with optional template support"""
        template = kw.pop("template", None)
        if template:
            return [prompt_template(template), generate(**kw)]
        return [generate(**kw)]
    
    builtin_solvers = {
        "multiple_choice": lambda **kw: [multiple_choice(**kw)],
        "generate": _generate_with_template,
        # SWE-bench solver
        "swebench_patch_solver": lambda **kw: [swebench_patch_solver(**kw)],
    }
    if solver_name in builtin_solvers:
        return builtin_solvers[solver_name](**solver_args)
    
    # Custom solvers
    try:
        import solvers as custom_solvers
        if hasattr(custom_solvers, solver_name):
            solver_fn = getattr(custom_solvers, solver_name)
            return [solver_fn(**solver_args)]
    except ImportError:
        pass
    
    raise ValueError(f"Unknown solver: {solver_name}. Available: {list(builtin_solvers.keys())}")


# =============================================================================
# Benchmark Creation
# =============================================================================

def create_benchmark(
    name: str,
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
    use_native_tools: bool | None = None,
    **kwargs,
) -> Task:
    """
    Create benchmark Task based on config
    
    If base exists: Get original Task and override only dataset
    If no base: Create new Task with config's solver/scorer
    
    Args:
        name: Benchmark name
        shuffle: Whether to shuffle data
        limit: Sample count limit
        split: Data split
        use_native_tools: Whether to use tool calling (used for BFCL, etc.)
                         None uses default (native)
    """
    config = get_benchmark_config(name)
    
    # Dynamic solver selection (BFCL, etc.)
    if use_native_tools is not None and config.get("metadata", {}).get("supports_dynamic_solver"):
        if use_native_tools:
            config["solver"] = config.get("metadata", {}).get("native_solver", config.get("solver"))
        else:
            config["solver"] = config.get("metadata", {}).get("text_solver", config.get("solver"))
    base_path = config.get("base", "")
    
    # =========================================================================
    # 1. Load data
    # =========================================================================
    data_type = config["data_type"]
    data_source = config["data_source"]
    
    if data_type == "weave":
        data_split = split or config.get("split")
        rows = load_weave_data(ref=data_source, split=data_split)
    else:
        rows = load_jsonl_data(data_source)
    
    # Sampling (before limit is applied)
    # sampling: "stratified" (maintain ratio) | "balanced" (equal) | None (slice from start)
    # sampling_by: Field name for grouping (e.g., "category")
    sampling = config.get("sampling")
    sampling_by = config.get("sampling_by")
    
    # Handle limit passed as string "None"
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
    
    # Add default values for missing fields
    default_fields = config.get("default_fields", {})
    if default_fields:
        for row in rows:
            for field, default_value in default_fields.items():
                if field not in row:
                    row[field] = default_value
    
    # =========================================================================
    # 2. If base exists: Override only original Task's dataset
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
        
        # Always apply system message if configured
        if config.get("system_message"):
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
    # 3. If no base: Create new Task with config's solver/scorer
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
        
        # Solver (with optional solver_args from config)
        solver_args = config.get("solver_args", {})
        if config.get("solver"):
            solver = get_solver_by_name(config["solver"], solver_args)
        else:
            solver = get_solver_by_name("multiple_choice", solver_args)
        
        # Always apply system message if configured
        if config.get("system_message"):
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
