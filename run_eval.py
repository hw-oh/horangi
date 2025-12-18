#!/usr/bin/env python3

import argparse
import locale
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

# Set English locale to fix inspect_evals date parsing issue
try:
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "C")
    except locale.Error:
        pass  # Continue even if locale setting fails

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import wandb
import weave
from inspect_ai import eval as inspect_eval
from core.config_loader import get_config

# All benchmarks list (active ones only)
ALL_BENCHMARKS = [
    "ko_hellaswag",
    "ko_aime2025",
    "ifeval_ko",
    # ko_balt_700: 구문해석/의미해석 분리 실행
    "ko_balt_700_syntax",
    "ko_balt_700_semantic",
    # haerae_bench_v1: 의미해석(RC)/일반지식(wo_RC) 분리 실행
    "haerae_bench_v1_rc",
    "haerae_bench_v1_wo_rc",
    "kmmlu",
    "kmmlu_pro",
    "squad_kor_v1",
    "ko_truthful_qa",
    "ko_moral",
    "ko_arc_agi",
    "hrm8k",  # HRM8K: 한국어 수학 추론 (GSM8K, KSM, MATH, MMMLU, OMNI_MATH 통합)
    "korean_hate_speech",
    "kobbq",
    "ko_hle",
    "ko_hallulens_wikiqa",
    # "ko_hallulens_longwiki",
    "ko_hallulens_nonexistent",
    "bfcl",
    "ko_mtbench",
    "swebench_verified_official_80",
]

# Quick test benchmarks (lightweight ones only)
QUICK_BENCHMARKS = [
    "ko_hellaswag",
    "kmmlu",
    "kobbq",
    "korean_hate_speech",
    "ifeval_ko",
    "ko_moral",
]


def get_model_env(config_name: str) -> dict[str, str]:
    """
    Generate API environment variables from model config file
    
    From configs/models/<config_name>.yaml:
    - base_url → OPENAI_BASE_URL (or provider-specific environment variable)
    - api_key_env → Read API key from that environment variable
    
    Returns:
        Environment variable dictionary
    """
    config = get_config()
    model_config = config.get_model(config_name)
    
    if not model_config:
        return {}
    
    env = {}
    
    # Check provider (based on model_id: openai/solar-pro2 → openai)
    model_id = model_config.get("model_id") or config_name
    provider = model_id.split("/")[0] if "/" in model_id else "openai"
    provider_upper = provider.upper()
    
    # Set base URL
    base_url = model_config.get("base_url") or model_config.get("api_base")
    if base_url:
        # OpenAI-compatible APIs use OPENAI_BASE_URL
        if provider in ["openai", "together", "groq", "fireworks"]:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env[f"{provider_upper}_BASE_URL"] = base_url
    
    # Set API key
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            # OpenAI-compatible APIs
            if provider in ["openai", "together", "groq", "fireworks"]:
                env["OPENAI_API_KEY"] = api_key
            else:
                env[f"{provider_upper}_API_KEY"] = api_key
    
    return env


def get_model_metadata(model: str) -> dict:
    """
    Load metadata from model config file
    
    Returns:
        {
            "release_date": "2024-05-13",
            "size_category": "flagship",
            "model_size": "unknown",
            ...
        }
    """
    config = get_config()
    model_config = config.get_model(model)
    
    if not model_config:
        return {}
    
    metadata = model_config.get("metadata", {})
    return {
        "release_date": metadata.get("release_date", "unknown"),
        "size_category": metadata.get("size_category", "unknown"),
        "model_size": metadata.get("model_size") or metadata.get("parameters", "unknown"),
    }


def get_task_function(benchmark: str):
    """
    Get task function from horangi.py by benchmark name
    
    Args:
        benchmark: Benchmark name (e.g., "ko_hellaswag")
    
    Returns:
        Task function
    """
    import horangi
    if hasattr(horangi, benchmark):
        return getattr(horangi, benchmark)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def get_inspect_model(config_name: str) -> tuple[str, dict]:
    """
    Get Inspect AI model string and model_args from config
    
    Returns:
        (model_string, model_args)
        e.g., ("openai/gpt-4o", {}) or ("openai/solar-pro2", {"api_key": "...", "base_url": "..."})
    """
    config = get_config()
    model_config = config.get_model(config_name)
    
    if not model_config:
        raise ValueError(f"Model config not found: {config_name}")
    
    model_id = model_config.get("model_id", config_name)
    api_provider = model_config.get("api_provider")
    
    # Determine Inspect model string
    if api_provider:
        model_name = model_id.split("/")[-1]
        inspect_model = f"{api_provider}/{model_name}"
    else:
        inspect_model = model_id
    
    # Build model_args for OpenAI-compatible APIs (non-official endpoints only)
    model_args = {}
    base_url = model_config.get("base_url") or model_config.get("api_base")
    
    # Skip base_url for official OpenAI API (inspect_ai already has it as default)
    if base_url and "api.openai.com" not in base_url:
        model_args["base_url"] = base_url
    
    # Only pass api_key for non-OpenAI official APIs (OpenAI uses OPENAI_API_KEY env var)
    api_key_env = model_config.get("api_key_env")
    if api_key_env and api_key_env != "OPENAI_API_KEY":
        api_key = os.environ.get(api_key_env)
        if api_key:
            model_args["api_key"] = api_key
    
    # Set INSPECT_WANDB_MODEL_NAME for Weave display
    os.environ["INSPECT_WANDB_MODEL_NAME"] = model_id
    
    return inspect_model, model_args


def get_model_generate_config(config_name: str, benchmark: str) -> dict:
    """
    Get generation config (temperature, max_tokens, etc.) from model config
    
    Returns:
        Generation config dict
    """
    config = get_config()
    model_config = config.get_model(config_name)
    
    if not model_config:
        return {}
    
    defaults = model_config.get("defaults", {})
    benchmark_overrides = model_config.get("benchmarks", {}).get(benchmark, {})
    
    # Merge defaults with benchmark-specific overrides
    generate_config = {}
    
    # Map config keys to inspect_ai.eval kwargs
    key_mapping = {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
    }
    
    for key, eval_key in key_mapping.items():
        if key in benchmark_overrides:
            generate_config[eval_key] = benchmark_overrides[key]
        elif key in defaults:
            generate_config[eval_key] = defaults[key]
    
    return generate_config


def run_benchmark(
    benchmark: str, 
    config_name: str,
    limit: int | None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> tuple[str, bool, str, dict | None]:
    """
    Run a single benchmark (in-process using inspect_ai.eval)
    
    Runs in the same process as the W&B run, so Weave traces are automatically
    linked to the run page.
    
    Returns:
        (benchmark_name, success, error_message, scores)
    """
    print(f"\n{'='*60}")
    print(f"🏃 Running: {benchmark}")
    print(f"{'='*60}")
    
    try:
        # Get task function
        task_fn = get_task_function(benchmark)
        
        # Create task with limit
        task = task_fn(limit=limit) if limit else task_fn()
        
        # Get model info
        inspect_model, model_args = get_inspect_model(config_name)
        
        # Get generation config
        generate_config = get_model_generate_config(config_name, benchmark)
        
        # Run evaluation using inspect_ai.eval()
        # This runs in the same process, so Weave traces are linked to the current W&B run
        eval_logs = inspect_eval(
            tasks=[task],
            model=inspect_model,
            model_args=model_args,
            limit=limit,
            log_dir="./logs",
            **generate_config,
        )
        
        if not eval_logs:
            return benchmark, False, "No evaluation logs returned", None
        
        eval_log = eval_logs[0]
        
        # Check success
        success = eval_log.status == "success"
        error_msg = "" if success else f"Status: {eval_log.status}"
        
        # Parse scores from eval_log
        scores = parse_scores_from_eval_log(eval_log, benchmark)
        
        return benchmark, success, error_msg, scores
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return benchmark, False, str(e), None


def parse_scores_from_eval_log(eval_log, benchmark: str) -> dict | None:
    """
    Parse scores from EvalLog object
    
    Returns:
        {"score": main_score, "details": {metric_name: value, ...}}
    """
    if not eval_log.results or not eval_log.results.scores:
        return None
    
    all_metrics = {}
    
    # Extract metrics from all scorers
    for score in eval_log.results.scores:
        scorer_name = score.name
        if score.metrics:
            for metric_name, metric in score.metrics.items():
                # Use scorer_name/metric_name format for clarity
                full_name = f"{metric_name}" if scorer_name == metric_name else f"{scorer_name}_{metric_name}"
                if hasattr(metric, 'value') and metric.value is not None:
                    all_metrics[full_name] = metric.value
    
    if not all_metrics:
        return None
    
    def find_metric(*suffixes: str) -> float | None:
        """Find metric by suffix (e.g., '_accuracy', '_avg')"""
        for suffix in suffixes:
            for key, value in all_metrics.items():
                if key.endswith(suffix) or key == suffix:
                    return value
        return None
    
    # Select main score based on benchmark
    main_score = None
    
    if benchmark == "ifeval_ko":
        # instruction_following_final_acc or instruction_following_prompt_strict_acc
        main_score = find_metric("_final_acc", "_prompt_strict_acc")
    elif benchmark == "kobbq":
        # kobbq_scorer_kobbq_avg
        main_score = find_metric("_kobbq_avg", "kobbq_avg")
    elif benchmark == "ko_hle":
        # hle_grader_hle_accuracy
        main_score = find_metric("_hle_accuracy", "hle_accuracy")
    elif "hallulens" in benchmark:
        # hallulens_qa_correct_rate or hallulens_refusal_refusal_rate
        main_score = find_metric("_correct_rate", "_refusal_rate")
    elif benchmark == "ko_mtbench":
        # mtbench_scorer_mean
        main_score = find_metric("_mean", "mean")
        if main_score is not None:
            main_score = main_score / 10.0  # Normalize to 0-1
    elif benchmark == "bfcl":
        # bfcl_scorer_accuracy
        main_score = find_metric("bfcl_scorer_accuracy", "_accuracy")
    elif benchmark == "squad_kor_v1":
        # f1_mean
        main_score = find_metric("f1_mean", "_f1_mean")
    elif benchmark == "swebench_verified_official_80":
        # swebench: resolved rate
        main_score = find_metric("_resolved", "resolved")
    
    # Fallback: find any accuracy metric
    if main_score is None:
        # Try common metric suffixes
        for suffix in ["_accuracy", "accuracy", "_mean", "mean", "_f1", "f1", "_resolved", "resolved"]:
            for key, value in all_metrics.items():
                if key.endswith(suffix):
                    main_score = value
                    # Normalize mtbench mean
                    if benchmark == "ko_mtbench" and suffix in ["_mean", "mean"]:
                        main_score = main_score / 10.0
                    break
            if main_score is not None:
                break
    
    return {
        "score": main_score,
        "details": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks and create leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default run (entity/project loaded from configs/base_config.yaml)
    uv run python run_eval.py --config gpt-4o

    # Quick test (lightweight benchmarks only)
    uv run python run_eval.py --config gpt-4o --quick
    
    # Run specific benchmarks only
    uv run python run_eval.py --config gpt-4o --only ko_hellaswag,kmmlu
"""
    )
    
    # Basic options
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config name (configs/models/<name>.yaml, e.g., gpt-4o, solar_pro2)",
    )
    parser.add_argument("--limit", type=int,
                        help="Number of samples per benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run only quick/light benchmarks")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated list of benchmarks to run (exclusive)")
    
    args = parser.parse_args()
    
    # Load W&B settings from environment variables (.env file already loaded)
    entity = os.environ.get("WANDB_ENTITY", "")
    project = os.environ.get("WANDB_PROJECT", "")
    
    if not entity or not project:
        print("❌ WANDB_ENTITY and WANDB_PROJECT environment variables are required for W&B logging.")
        print("   Add the following to your .env file:")
        print("     WANDB_ENTITY=your-entity")
        print("     WANDB_PROJECT=your-project")
        sys.exit(1)
    
    config = get_config()
    
    # Filter benchmarks
    if args.quick:
        benchmarks = QUICK_BENCHMARKS
    elif args.only:
        benchmarks = [b.strip() for b in args.only.split(",") if b.strip()]
        # Validate
        invalid = [b for b in benchmarks if b not in ALL_BENCHMARKS]
        if invalid:
            print(f"❌ Unknown benchmarks: {invalid}")
            print(f"   Available: {ALL_BENCHMARKS}")
            sys.exit(1)
    else:
        benchmarks = ALL_BENCHMARKS
    
    # Load model config (configs/models/<name>.yaml)
    model_cfg = config.get_model(args.config)
    if not model_cfg:
        print(f"❌ Model configuration not found: {args.config}")
        print("   Check if YAML file exists in configs/models/ directory.")
        sys.exit(1)

    model_id = model_cfg.get("model_id") or args.config

    # Display model name (openai/solar-pro2 → solar-pro2)
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    
    wandb_run = wandb.init(
        entity=entity,
        project=project,
        name=model_name,
        job_type="evaluation",
        config={
            "config": args.config,
            "model": model_id,
            "model_name": model_name,
            "limit": args.limit,
            "benchmarks": benchmarks,
        },
    )
    print(f"✅ W&B run started: {wandb_run.url}")
    
    # Initialize Weave in the same process - this links all Weave traces to the W&B run
    weave.init(f"{entity}/{project}")
    print(f"✅ Weave initialized: {entity}/{project}")
    
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model: {model_id}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    print(f"Leaderboard: {entity}/{project}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Track execution results
    results = []
    benchmark_scores = {}
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] ", end="")
        name, success, error, scores = run_benchmark(
            benchmark, 
            args.config,
            args.limit,
            wandb_entity=entity,
            wandb_project=project,
        )
        results.append((name, success, error))
        
        if scores:
            benchmark_scores[name] = scores
    
    # Results summary
    print(f"\n\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\n✅ Successful: {len(successful)} / {len(results)}")
    for name, _, _ in successful:
        score_info = benchmark_scores.get(name, {})
        score_str = f" (score: {score_info.get('score', 'N/A')})" if score_info else ""
        print(f"   - {name}{score_str}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} / {len(results)}")
        for name, _, error in failed:
            print(f"   - {name}: {error}")
    
    # Print detailed results table by category
    print(f"\n{'='*60}")
    print(f"📋 Detailed Results by Category")
    print(f"{'='*60}")
    
    for benchmark_name, score_info in benchmark_scores.items():
        details = score_info.get("details", {})
        if len(details) > 1:  # Only if there are detailed results
            print(f"\n📌 {benchmark_name}")
            print(f"   {'─'*40}")
            
            # Separate main metrics and category metrics
            main_metrics = []
            category_metrics = []
            
            for metric, value in sorted(details.items()):
                if "_score" in metric or "_accuracy" in metric or "_rate" in metric or "_acc" in metric:
                    category_metrics.append((metric, value))
                else:
                    main_metrics.append((metric, value))
            
            # Print main metrics
            for metric, value in main_metrics:
                print(f"   {metric:<30} {value:.4f}")
            
            # Print category metrics (table format)
            if category_metrics:
                print(f"   {'─'*40}")
                for metric, value in category_metrics:
                    print(f"   {metric:<30} {value:.4f}")
    
    # Create Weave Leaderboard (if there are successful benchmarks)
    if benchmark_scores and entity and project:
        try:
            from core.weave_leaderboard import create_weave_leaderboard
            # Pass only successful benchmark list
            successful_benchmarks = list(benchmark_scores.keys())
            leaderboard_url = create_weave_leaderboard(
                entity=entity,
                project=project,
                benchmarks=successful_benchmarks,
            )
            if leaderboard_url:
                print(f"\n🏆 Leaderboard URL: {leaderboard_url}")
        except Exception as e:
            print(f"⚠️ Weave Leaderboard creation failed: {e}")
    
    # Log W&B Leaderboard Tables (if there are successful benchmarks)
    if benchmark_scores and wandb_run is not None:
        try:
            from core.models_leaderboard import log_leaderboard_tables
            
            print(f"\n{'='*60}")
            print(f"📊 Logging W&B Leaderboard Tables")
            print(f"{'='*60}")
            
            # Get model metadata
            model_metadata = get_model_metadata(args.config)
            
            log_leaderboard_tables(
                wandb_run=wandb_run,
                model_name=model_name,
                benchmark_scores=benchmark_scores,
                metadata=model_metadata,
            )
        except Exception as e:
            import traceback
            print(f"⚠️ W&B Leaderboard table creation failed: {e}")
            traceback.print_exc()
    
    # End W&B run
    if wandb_run is not None:
        print(f"\n📊 Ending W&B run...")
        wandb_run.finish()
        print(f"✅ W&B run completed!")
    
    print(f"\n{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Exit code 0 even if some failures, as long as leaderboard was created
    # (partial benchmark failures still have results saved)
    sys.exit(0)


if __name__ == "__main__":
    main()
