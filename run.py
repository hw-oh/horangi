#!/usr/bin/env python3
"""
Batch evaluation runner with automatic GPU allocation.

For vLLM models: Uses srun with GPU count from config (vllm.tensor_parallel_size)
For API models: Runs directly without GPU allocation
"""

import os
import sys
from pathlib import Path

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.config_loader import get_config


# Model configurations by provider
lgai_configs = [
    # "EXAONE-4.0.1-32B",
    # "EXAONE-4.0-1.2B",
    # "EXAONE-4.0-32B",
    # "EXAONE-3.5-32B-Instruct",
    # "EXAONE-3.5-7.8B-Instruct",
    # "EXAONE-3.5-2.4B-Instruct",
    # "EXAONE-3.0-7.8B-Instruct",
]

openai_configs = [
    # "gpt-4o-2024-11-20",
    # # "gpt-4.1-2025-04-14",
    # "gpt-5-nano-2025-08-07_high-effort",
    # "gpt-5-mini-2025-08-07_high-effort",
    # "gpt-5-2025-08-07_high-effort",
    "gpt-5.2-2025-12-11_xhigh-effort",
    "gpt-5.1-2025-11-13_high-effort",
    "o4-mini-2025-04-16",
]

anthropic_configs = [
    # "claude-opus-4-5-20251101_low-effort",
    # "claude-opus-4-5-20251101_high-effort",
    # "claude-haiku-4-5-20251001_high-effort",
    # "claude-sonnet-4-5-20250929_high-effort",
    "claude-opus-4-20250514_high-effort",
    "claude-opus-4-1-20250805_high-effort",
]

xai_configs = [
    "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-reasoning",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "grok-4-0709",
]

google_configs = [
    # "gemini-2.5-flash-lite",✅
    # "gemini-2.5-flash-lite_high-effort",✅
    # "gemini-2.5-flash",✅
    # "gemini-2.5-flash_high-effort",✅
    # "gemini-2.5-pro_low-effort",
    # "gemini-2.5-pro_high-effort",✅
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview_high-effort",
    "gemini-3-pro-preview_high-effort",
]

together_configs = [
    # "gpt-oss-20b",
    # "gpt-oss-120b",
    # "Kimi-K2-Thinking",
    "Kimi-K2-Instruct-0905",
    "DeepSeek-R1",
    "DeepSeek-V3",
    "DeepSeek-V3.1",
]

gemma_configs = [
    "gemma-3-270m-it",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]

qwen_configs = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Qwen3-4B-Instruct-2507",
]


def get_gpu_count(config_name: str) -> int:
    """
    Get required GPU count from config file.
    
    Returns:
        GPU count from vllm.tensor_parallel_size, or 0 for API models
    """
    config = get_config()
    vllm_config = config.get_vllm_config(config_name)
    
    if vllm_config:
        return vllm_config.get("tensor_parallel_size", 1)
    else:
        # API model, no GPU needed
        return 0


def run_eval(config_name: str, extra_args: str = "", time_limit: str = "8:00:00") -> int:
    """
    Run evaluation for a single model config.
    
    Args:
        config_name: Model config name
        extra_args: Additional arguments for run_eval.py (e.g., "--only ko_hellaswag --limit 10")
        time_limit: Time limit for srun (default: 8 hours)
    
    Returns:
        Exit code from the command
    """
    gpu_count = get_gpu_count(config_name)
    
    base_cmd = f"cd ~/workspace/horangi && uv run python run_eval.py --config {config_name}"
    if extra_args:
        base_cmd += f" {extra_args}"
    
    if gpu_count > 0:
        # vLLM model: use srun with GPU allocation
        cmd = f'srun --gres=gpu:{gpu_count} --time={time_limit} bash -c "{base_cmd}"'
        print(f"\n🖥️  Running with {gpu_count} GPU(s): {config_name}")
    else:
        # API model: run directly
        cmd = base_cmd
        print(f"\n☁️  Running API model: {config_name}")
    
    print(f"   Command: {cmd}\n")
    return os.system(cmd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner with automatic GPU allocation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all LGAI models
    python run.py --provider lgai
    
    # Run all API models (OpenAI, Anthropic, etc.)
    python run.py --provider openai
    
    # Quick test with limited samples
    python run.py --provider lgai --only ko_hellaswag --limit 10
    
    # Run specific benchmarks
    python run.py --provider lgai --only ko_hellaswag,kmmlu
"""
    )
    parser.add_argument("--provider", type=str, default=None,
                        choices=["openai", "anthropic", "xai", "google", "lgai", "together", "qwen", "gemma", "all"],
                        help="Provider to run evaluations for")
    parser.add_argument("--only", type=str, default="",
                        help="Run only specific benchmarks (comma-separated)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit samples per benchmark")
    parser.add_argument("--time", type=str, default="8:00:00",
                        help="Time limit for srun (default: 8:00:00)")
    
    args = parser.parse_args()

    # Select configs based on provider
    if args.provider == 'openai':
        configs = openai_configs
    elif args.provider == 'anthropic':
        configs = anthropic_configs
    elif args.provider == 'xai':
        configs = xai_configs
    elif args.provider == 'google':
        configs = google_configs
    elif args.provider == 'lgai':
        configs = lgai_configs
    elif args.provider == 'together':
        configs = together_configs
    elif args.provider == 'qwen':
        configs = qwen_configs
    elif args.provider == 'gemma':
        configs = gemma_configs
    elif args.provider == 'all':
        configs = lgai_configs + openai_configs + anthropic_configs + xai_configs + google_configs + together_configs + vllm_configs
    else:
        configs = lgai_configs + openai_configs + anthropic_configs + xai_configs + google_configs + together_configs
    
    # Build extra args
    extra_args = ""
    if args.only:
        extra_args += f"--only {args.only}"
    if args.limit:
        extra_args += f" --limit {args.limit}"
    extra_args = extra_args.strip()
    
    print("\n" + "="*60)
    print("🐯 Horangi Batch Evaluation Runner")
    print("="*60)
    print(f"Provider: {args.provider or 'default'}")
    print(f"Models: {len(configs)}")
    print(f"Time limit: {args.time}")
    if extra_args:
        print(f"Extra args: {extra_args}")
    print("="*60)
    
    # Run evaluations
    results = []
    for config in configs:
        exit_code = run_eval(config, extra_args=extra_args, time_limit=args.time)
        results.append((config, exit_code))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    success = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]
    
    print(f"✅ Success: {len(success)} / {len(results)}")
    for config, _ in success:
        print(f"   - {config}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} / {len(results)}")
        for config, code in failed:
            print(f"   - {config} (exit code: {code})")
