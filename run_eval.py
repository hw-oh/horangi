#!/usr/bin/env python3

import argparse
import locale
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (override=True ensures .env values take precedence over existing env vars)
load_dotenv(Path(__file__).parent / ".env", override=True)

# Patch LLM clients EARLY for Weave token tracking (must be before clients are imported)
try:
    from weave.integrations.anthropic import anthropic_sdk
    anthropic_sdk.get_anthropic_patcher().attempt_patch()
except Exception:
    pass

try:
    from weave.integrations.litellm import litellm as weave_litellm
    weave_litellm.get_litellm_patcher().attempt_patch()
except Exception:
    pass

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

import pandas as pd
import wandb
from inspect_ai import eval as inspect_eval
from core.config_loader import get_config
from server.vllm_manager import VLLMServerManager

# Register custom model providers
try:
    import providers  # noqa: F401 - registers litellm provider
except ImportError:
    pass

# All benchmarks list (active ones only)
ALL_BENCHMARKS = [
    "ko_arc_agi",
    "kmmlu_pro",
    "ko_hle",
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
    "squad_kor_v1",
    "ko_truthful_qa",
    "ko_moral",
    "hrm8k",
    "korean_hate_speech",
    "kobbq",
    "ko_mtbench",
    "ko_hallulens_wikiqa",
    # "ko_hallulens_longwiki",
    "ko_hallulens_nonexistent",
    "bfcl",
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
    
    From new config structure (model.base_url, model.api_key_env):
    - base_url → OPENAI_BASE_URL (or provider-specific environment variable)
    - api_key_env → Read API key from that environment variable
    
    Returns:
        Environment variable dictionary
    """
    config = get_config()
    
    env = {}
    
    # Get client type and provider
    client = config.get_model_client(config_name)
    provider = config.get_model_provider(config_name) or "openai"
    provider_upper = provider.upper()
    
    # Set base URL
    base_url = config.get_model_base_url(config_name)
    if base_url:
        # hosted_vllm은 HOSTED_VLLM_API_BASE 사용
        if provider == "hosted_vllm":
            env["HOSTED_VLLM_API_BASE"] = base_url
        # OpenAI-compatible APIs use OPENAI_BASE_URL
        elif client == "openai" or provider in ["openai", "together", "groq", "fireworks"]:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env[f"{provider_upper}_BASE_URL"] = base_url
    
    # Set API key
    api_key_env = config.get_model_api_key_env(config_name)
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            # hosted_vllm은 HOSTED_VLLM_API_KEY 사용
            if provider == "hosted_vllm":
                env["HOSTED_VLLM_API_KEY"] = api_key
            # OpenAI-compatible APIs
            elif client == "openai" or provider in ["openai", "together", "groq", "fireworks"]:
                env["OPENAI_API_KEY"] = api_key
            else:
                env[f"{provider_upper}_API_KEY"] = api_key
    
    return env


def get_model_metadata(config_name: str) -> dict:
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
    metadata = config.get_metadata(config_name)
    
    return {
        "release_date": metadata.get("release_date", "unknown"),
        "size_category": metadata.get("size_category", "unknown"),
        "model_size": metadata.get("model_size") or metadata.get("parameters", "unknown"),
    }


def get_previous_benchmark_scores(entity: str, project: str, run_id: str) -> dict:
    """
    W&B API로 이전 run의 benchmark_detail_table에서 점수 불러오기
    
    Args:
        entity: W&B entity
        project: W&B project
        run_id: W&B run ID
    
    Returns:
        {"benchmark_name": {"score": 0.85, "details": {...}}, ...}
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # benchmark_detail_table에서 결과 추출
        # W&B Table은 run.history()에서 가져올 수 있음
        history = run.history(keys=["benchmark_detail_table"])
        
        if history.empty or "benchmark_detail_table" not in history.columns:
            print(f"⚠️ benchmark_detail_table not found in run {run_id}")
            return {}
        
        # 마지막 로깅된 테이블 가져오기
        table_data = history["benchmark_detail_table"].dropna().iloc[-1]
        
        if table_data is None:
            print(f"⚠️ benchmark_detail_table is empty in run {run_id}")
            return {}
        
        # W&B Table JSON에서 데이터 추출
        # table_data는 wandb.Table 참조이므로 artifact에서 가져와야 함
        benchmark_scores = {}
        
        # run.summary에서 직접 가져오기 시도 (더 간단한 방법)
        # 또는 logged artifacts에서 테이블 데이터 가져오기
        for artifact in run.logged_artifacts():
            if "benchmark_detail_table" in artifact.name:
                table = artifact.get("benchmark_detail_table")
                if table:
                    df = table.get_dataframe()
                    for _, row in df.iterrows():
                        benchmark_name = row.get("benchmark")
                        if benchmark_name:
                            score_info = {"score": row.get("score")}
                            # detail_ 접두사가 붙은 컬럼들을 details로 수집
                            details = {}
                            for col in df.columns:
                                if col.startswith("detail_"):
                                    detail_key = col.replace("detail_", "")
                                    if pd.notna(row.get(col)):
                                        details[detail_key] = row.get(col)
                            if details:
                                score_info["details"] = details
                            benchmark_scores[benchmark_name] = score_info
                    break
        
        # Artifact에서 못 가져온 경우, summary에서 개별 점수 가져오기 시도
        if not benchmark_scores:
            summary = run.summary
            # summary에 저장된 벤치마크별 점수가 있는지 확인
            for key, value in summary.items():
                # 벤치마크 이름 패턴 매칭 (예: ko_arc_agi_score)
                if key.endswith("_score") and isinstance(value, (int, float)):
                    benchmark_name = key.replace("_score", "")
                    benchmark_scores[benchmark_name] = {"score": value, "details": {}}
        
        print(f"✅ Loaded {len(benchmark_scores)} benchmark scores from previous run")
        return benchmark_scores
        
    except Exception as e:
        print(f"⚠️ Failed to load previous benchmark scores: {e}")
        import traceback
        traceback.print_exc()
        return {}


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


def get_inspect_model(config_name: str, benchmark: str | None = None) -> tuple[str, dict, str | None]:
    """
    Get Inspect AI model string, model_args, and base_url from config
    
    Returns:
        (model_string, model_args, base_url)
        e.g., ("litellm/anthropic/claude-opus-4-5-20251101", {}, None)
              ("openai/solar-pro2-251215", {"api_key": "..."}, "https://...")
    """
    config = get_config()
    
    # Build model string using new config structure
    inspect_model = config.get_inspect_model_string(config_name)
    
    # Get client type and provider
    client = config.get_model_client(config_name)
    provider = config.get_model_provider(config_name)
    
    # Get base_url (for OpenAI-compatible APIs)
    base_url = config.get_model_base_url(config_name)
    
    # For litellm hosted_vllm: use environment variable instead of model_base_url
    if client == "litellm" and provider == "hosted_vllm":
        if base_url:
            os.environ["HOSTED_VLLM_API_BASE"] = base_url
        # Get API key from config and set environment variable
        api_key_env = config.get_model_api_key_env(config_name)
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                os.environ["HOSTED_VLLM_API_KEY"] = api_key
        base_url = None  # Don't pass to inspect_eval
    # Skip base_url for official OpenAI API (inspect_ai already has it as default)
    elif base_url and "api.openai.com" in base_url:
        base_url = None
    
    # Build model_args
    model_args = config.get_inspect_model_args(config_name, benchmark)
    
    # Set INSPECT_WANDB_MODEL_NAME for Weave display
    model_name = config.get_model_name(config_name)
    os.environ["INSPECT_WANDB_MODEL_NAME"] = model_name
    
    return inspect_model, model_args, base_url


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dicts. Override values take precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)
    
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_model_generate_config(config_name: str, benchmark: str) -> dict:
    """
    Get generation config (temperature, max_tokens, etc.) from model config
    
    Returns:
        Generation config dict
    """
    config = get_config()
    
    # Get model params (new structure: model.params)
    params = config.get_model_params(config_name)
    
    # Get benchmark-specific overrides
    benchmark_overrides = config.get_benchmark_config(config_name, benchmark)
    
    # Merge params with benchmark-specific overrides
    generate_config = {}
    
    # Map config keys to inspect_ai.eval kwargs
    key_mapping = {
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "effort": "effort",  # Anthropic
        "reasoning_effort": "reasoning_effort",  # OpenAI
        "reasoning_tokens": "reasoning_tokens",
        "timeout": "timeout",
        "max_retries": "max_retries",
    }
    
    for key, eval_key in key_mapping.items():
        if key in benchmark_overrides:
            generate_config[eval_key] = benchmark_overrides[key]
        elif key in params:
            generate_config[eval_key] = params[key]
    
    # extra_body: litellm completion 호출 시 추가 파라미터 (GLM enable_thinking 등)
    # Deep merge로 전역 config를 기본으로 하고, 벤치마크별 config가 겹치는 부분만 덮어쓰기
    base_extra_body = params.get("extra_body", {})
    override_extra_body = benchmark_overrides.get("extra_body", {})
    extra_body = deep_merge(base_extra_body, override_extra_body)
    if extra_body:
        generate_config["extra_body"] = extra_body
    
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
        inspect_model, model_args, base_url = get_inspect_model(config_name, benchmark)
        
        # Get generation config
        generate_config = get_model_generate_config(config_name, benchmark)
        
        # Run evaluation using inspect_ai.eval()
        # This runs in the same process, so Weave traces are linked to the current W&B run
        #
        # fail_on_error=False + continue_on_fail=True:
        #   샘플 에러(500/RetryError 등)가 나도 전체 eval을 중단하지 않고,
        #   해당 샘플만 error(=틀림) 처리 후 나머지 샘플 계속 진행
        eval_logs = inspect_eval(
            tasks=[task],
            model=inspect_model,
            model_args=model_args,
            model_base_url=base_url,
            limit=limit,
            log_dir="./logs",
            fail_on_error=False,
            continue_on_fail=True,
            display="plain",  # tqdm-style progress bar
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
    parser.add_argument("--tag", type=str, action="append", default=[],
                        help="Additional W&B tags (can be used multiple times, e.g., --tag exp1 --tag test)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume existing W&B run by run ID (e.g., abc123xy)")
    
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

    # Get model name from new config structure
    model_name = config.get_model_name(args.config)
    
    # Get W&B run name from config (or fallback to model_name)
    wandb_run_name = config.get_wandb_run_name(args.config) or model_name

    # Combine default tags with user-provided tags
    tags = ["inspect"] + args.tag
    
    # Check for vLLM server auto-start configuration
    vllm_config = config.get_vllm_config(args.config)
    vllm_server = None
    
    if vllm_config:
        print(f"\n🔧 vLLM server configuration detected")
        print(f"   Model: {vllm_config.get('model_path')}")
        print(f"   Will auto-start server before evaluation")
        tags.append("vllm")
    
    # 이전 벤치마크 결과 (resume 시 사용)
    previous_benchmark_scores = {}
    
    if args.resume:
        # Resume 전에 이전 run의 벤치마크 결과 불러오기
        print(f"\n📥 Loading previous benchmark scores from run {args.resume}...")
        previous_benchmark_scores = get_previous_benchmark_scores(entity, project, args.resume)
        
        # Resume existing run
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            id=args.resume,
            resume="must",  # 반드시 기존 run이어야 함
        )
        print(f"✅ W&B run resumed: {wandb_run.url}")
    else:
        # Create new run
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            name=wandb_run_name,
            job_type="evaluation",
            tags=tags,
            config={
                "config": args.config,
                "model": model_name,
                "model_name": model_name,
                "limit": args.limit,
                "benchmarks": benchmarks,
            },
        )
        print(f"✅ W&B run started: {wandb_run.url}")
    
    # Note: Weave is initialized by inspect_wandb hooks automatically
    # Don't call weave.init() here as it conflicts with the hooks' initialization
    print(f"✅ Weave will be initialized by inspect_wandb hooks: {entity}/{project}")
    print("✅ Anthropic client patched for Weave token tracking (early patch)")
    
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model: {model_name}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    print(f"Leaderboard: {entity}/{project}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if vllm_config:
        print(f"vLLM Auto-Start: Enabled")
    print(f"{'='*60}")
    
    # Start vLLM server if configured
    if vllm_config:
        try:
            vllm_server = VLLMServerManager(vllm_config)
            vllm_server.start()
            
            # Update base_url in environment if using hosted_vllm
            # This ensures the model connects to the auto-started server
            os.environ["HOSTED_VLLM_API_BASE"] = vllm_server.base_url
        except Exception as e:
            print(f"❌ Failed to start vLLM server: {e}")
            if wandb_run is not None:
                wandb_run.finish(exit_code=1)
            sys.exit(1)
    
    # Track execution results
    results = []
    benchmark_scores = {}
    
    try:
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
    finally:
        # Stop vLLM server if we started it
        if vllm_server is not None:
            vllm_server.stop()
    
    # Resume인 경우 이전 결과와 merge (새 결과가 우선)
    if args.resume and previous_benchmark_scores:
        print(f"\n📦 Merging with previous benchmark scores...")
        print(f"   Previous: {len(previous_benchmark_scores)} benchmarks")
        print(f"   New: {len(benchmark_scores)} benchmarks")
        
        # 이전 결과를 기본으로 하고, 새 결과로 덮어씀
        merged_scores = {**previous_benchmark_scores, **benchmark_scores}
        benchmark_scores = merged_scores
        
        print(f"   Merged: {len(benchmark_scores)} benchmarks")
    
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
