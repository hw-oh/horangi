#!/usr/bin/env python
"""
Horangi CLI - Korean LLM Benchmark Evaluation Tool

Usage:
    uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5
    uv run horangi ko_hellaswag --config gpt-4o -T limit=5
    uv run horangi swebench_verified_official_80 --config claude-3-5-sonnet -T limit=1
    uv run horangi --list  # List available benchmarks
    uv run horangi --list-models  # List available model configurations
    uv run horangi leaderboard --project <entity>/<project>  # Create leaderboard
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file (project root)
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


def _ensure_wandb_env() -> bool:
    """
    Check and set WANDB_ENTITY and WANDB_PROJECT environment variables
    
    Prompts user for input if environment variables are not set.
    
    Returns:
        True if environment variables are set, False if user cancels
    """
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    
    if entity and project:
        return True
    
    print("⚠️  W&B environment variables are not set.")
    print()
    
    if not entity:
        try:
            entity = input("WANDB_ENTITY (team or username): ").strip()
            if not entity:
                print("❌ WANDB_ENTITY is required.")
                return False
            os.environ["WANDB_ENTITY"] = entity
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            return False
    
    if not project:
        try:
            project = input("WANDB_PROJECT (project name): ").strip()
            if not project:
                print("❌ WANDB_PROJECT is required.")
                return False
            os.environ["WANDB_PROJECT"] = project
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            return False
    
    print()
    print(f"✅ Project: {entity}/{project}")
    print()
    
    return True


def _is_openai_compat_api(model_config: dict) -> bool:
    """
    Check if the model uses OpenAI-compatible API
    
    Returns True if one of the following conditions is met:
    1. api_provider is "openai" and base_url is not openai.com
    2. model_id starts with "openai/" and base_url is not openai.com
    
    Examples: Solar, Grok, Together AI, etc.
    """
    api_provider = model_config.get("api_provider")
    model_id = model_config.get("model_id", "")
    base_url = model_config.get("base_url") or model_config.get("api_base")
    
    # api_provider is openai and base_url is not openai.com
    if api_provider == "openai" and base_url:
        return "openai.com" not in base_url
    
    # Legacy: openai/ provider with base_url not openai.com
    if model_id.startswith("openai/") and base_url:
        return "openai.com" not in base_url
    
    return False


def _get_openai_compat_args(model_config: dict) -> list[str]:
    """
    Generate CLI arguments for OpenAI-compatible API
    
    Uses api_key_env from model config instead of OPENAI_API_KEY from .env,
    passing it directly via --model-args api_key=...
    
    Returns:
        List of CLI arguments (e.g., ["--model-args", "api_key=...", "--model-base-url", "..."])
    """
    extra_args = []
    
    if not _is_openai_compat_api(model_config):
        return extra_args
    
    # API key: pass directly via -M api_key=... (bypass .env)
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            extra_args.extend(["-M", f"api_key={api_key}"])
        else:
            print(f"❌ Environment variable {api_key_env} is not set!")
            print(f"   Set it with: export {api_key_env}=\"your-api-key\"")
    
    # Base URL: pass via --model-base-url
    base_url = model_config.get("base_url") or model_config.get("api_base")
    if base_url:
        extra_args.extend(["--model-base-url", base_url])
    
    return extra_args


def _handle_leaderboard_command(args: list[str]) -> int:
    """
    Handle leaderboard creation command
    
    Usage:
        horangi leaderboard --project <entity>/<project>
        horangi leaderboard --project <entity>/<project> --name "My Leaderboard"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Weave leaderboard",
        prog="horangi leaderboard",
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Weave project (e.g., my-team/my-project)",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Leaderboard name (default: Korean LLM Leaderboard)",
    )
    parser.add_argument(
        "--description", "-d",
        default=None,
        help="Leaderboard description",
    )
    
    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return e.code if e.code else 0
    
    # Split entity and project from project string
    if "/" not in parsed.project:
        print("❌ Invalid project format. Use '<entity>/<project>' format.")
        return 1
    
    entity, project = parsed.project.split("/", 1)
    
    print(f"🐯 Horangi - Create Weave Leaderboard")
    print(f"📁 Project: {entity}/{project}")
    print()
    
    # Add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    
    from core.weave_leaderboard import (
        create_weave_leaderboard,
        LEADERBOARD_NAME,
        LEADERBOARD_DESCRIPTION,
    )
    
    name = parsed.name or LEADERBOARD_NAME
    description = parsed.description or LEADERBOARD_DESCRIPTION
    
    url = create_weave_leaderboard(
        name=name,
        description=description,
        entity=entity,
        project=project,
    )
    
    return 0 if url else 1


def main():
    args = sys.argv[1:]
    
    # Find project root (src/cli/__init__.py -> project root)
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    horangi_py = project_root / "horangi.py"
    
    # Add src to path (for config_loader, etc.)
    sys.path.insert(0, str(src_path))
    
    # leaderboard: Create leaderboard
    if args and args[0] == "leaderboard":
        return _handle_leaderboard_command(args[1:])
    
    # --list-models: Print model configuration list
    if args and args[0] == "--list-models":
        print("🐯 Horangi - Available Model Configurations")
        print()
        
        from core.config_loader import ConfigLoader
        config = ConfigLoader()
        models = config.list_models()
        
        if not models:
            print("  No models configured.")
            print(f"  Add YAML files to configs/models/ directory.")
        else:
            print("Available model configurations:")
            print()
            for model_name in sorted(models):
                if model_name.startswith("_"):  # Exclude template files
                    continue
                model_config = config.get_model(model_name)
                model_id = model_config.get("model_id", model_name)
                metadata = model_config.get("metadata", {})
                desc = metadata.get("description", "")
                release_date = metadata.get("release_date", "")
                
                print(f"  {model_name:<25} → {model_id}")
                if desc:
                    print(f"  {'':25}   {desc}")
                if release_date:
                    print(f"  {'':25}   Release date: {release_date}")
                print()
        
        print("Usage example:")
        print("  uv run horangi ko_hellaswag --config gpt-4o -T limit=5")
        return 0
    
    # --list or -l option: Print benchmark list
    if not args or args[0] in ("--list", "-l", "--help", "-h"):
        print("🐯 Horangi - Korean LLM Benchmark Evaluation Tool")
        print()
        print("Usage:")
        print("  uv run horangi <benchmark> --model <model> [options]")
        print("  uv run horangi <benchmark> --config <config_file> [options]")
        print()
        print("Examples:")
        print("  uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5")
        print("  uv run horangi ko_hellaswag --config gpt-4o -T limit=5")
        print("  uv run horangi swebench_verified_official_80 --config claude-3-5-sonnet -T limit=1")
        print()
        print("Model configuration list:")
        print("  uv run horangi --list-models")
        print()
        print("Create leaderboard:")
        print("  uv run horangi leaderboard --project <entity>/<project>")
        print()
        
        # Print benchmark list
        print("Available benchmarks:")
        print()
        
        from benchmarks import list_benchmarks_with_descriptions
        
        # Group by category
        categories = {
            "General": ["ko_hellaswag", "ko_aime2025", "ifeval_ko", "ko_balt_700"],
            "Knowledge": ["haerae_bench_v1_rc", "haerae_bench_v1_wo_rc", "kmmlu", "kmmlu_pro", "squad_kor_v1", "ko_truthful_qa"],
            "Reasoning": ["ko_moral", "ko_arc_agi", "hrm8k"],
            "Bias/Safety": ["korean_hate_speech", "kobbq", "ko_hle"],
            "Hallucination (HalluLens)": ["ko_hallulens_wikiqa", "ko_hallulens_longwiki", "ko_hallulens_generated", "ko_hallulens_mixed", "ko_hallulens_nonexistent"],
            "Function Calling": ["bfcl"],
            "Conversation": ["ko_mtbench"],
            "Coding": ["swebench_verified_official_80"],
        }
        
        benchmarks_dict = dict(list_benchmarks_with_descriptions())
        
        for category, names in categories.items():
            print(f"  [{category}]")
            for name in names:
                desc = benchmarks_dict.get(name, "")
                print(f"    {name:<35} {desc}")
            print()
        
        print(f"Total {len(benchmarks_dict)} benchmarks")
        return 0
    
    # First argument is benchmark name
    benchmark = args[0]
    rest_args = list(args[1:])
    
    # Process --config or -c option
    config_name = None
    new_args = []
    i = 0
    while i < len(rest_args):
        arg = rest_args[i]
        if arg in ("--config", "-c"):
            if i + 1 < len(rest_args):
                config_name = rest_args[i + 1]
                i += 2
                continue
            else:
                print("❌ --config option requires a model configuration name.")
                print("   Example: --config gpt-4o")
                return 1
        new_args.append(arg)
        i += 1
    
    rest_args = new_args
    
    # Load model info from configuration file
    if config_name:
        from core.config_loader import ConfigLoader
        
        config = ConfigLoader()
        model_config = config.get_model(config_name)
        
        if not model_config:
            print(f"❌ Model configuration not found: {config_name}")
            print(f"   Available models: {', '.join(config.list_models())}")
            return 1
        
        # Generate OpenAI-compatible API arguments (Solar, Grok, etc.)
        # Use api_key_env from model config instead of OPENAI_API_KEY from .env
        openai_compat_args = _get_openai_compat_args(model_config)
        
        # Handle model_id and api_provider
        # model_id: User-visible name (e.g., upstage/solar-pro2)
        # api_provider: Actual API provider (e.g., openai - for OpenAI-compatible APIs)
        model_id = model_config.get("model_id", config_name)
        api_provider = model_config.get("api_provider")
        
        if api_provider:
            # If api_provider is specified: upstage/solar-pro2 → openai/solar-pro2
            model_name = model_id.split("/")[-1]  # Extract model name only
            inspect_model = f"{api_provider}/{model_name}"
        else:
            inspect_model = model_id
        
        # Set model name for Weave display (used by inspect-wandb)
        # Use metadata.name if available, otherwise extract model name from model_id
        model_name_for_weave = model_config.get("metadata", {}).get("name") or (model_id.split("/")[-1] if "/" in model_id else model_id)
        os.environ["INSPECT_WANDB_MODEL_NAME"] = model_name_for_weave
        
        # Add --model if not already specified
        has_model = any(arg == "--model" for arg in rest_args)
        if not has_model:
            rest_args = ["--model", inspect_model] + rest_args
        
        # Apply benchmark-specific settings
        benchmark_overrides = model_config.get("benchmarks", {}).get(benchmark, {})
        defaults = model_config.get("defaults", {})
        
        # Apply settings (add via -T option, keep existing ones)
        existing_t_args = set()
        for j, arg in enumerate(rest_args):
            if arg == "-T" and j + 1 < len(rest_args):
                key = rest_args[j + 1].split("=")[0]
                existing_t_args.add(key)
        
        # Apply defaults
        # - generate_params: Parameters used in API requests (--temperature, --max-tokens, etc.)
        # - task_params (-T): Parameters passed to task function
        # Options directly supported by inspect eval CLI (convert to kebab-case)
        generate_param_mapping = {
            "temperature": "--temperature",
            "max_tokens": "--max-tokens",
            "top_p": "--top-p",
            "top_k": "--top-k",
            "stop": "--stop-seqs",
            "frequency_penalty": "--frequency-penalty",
            "presence_penalty": "--presence-penalty",
        }
        for key, value in defaults.items():
            if key not in existing_t_args:
                if key in generate_param_mapping:
                    rest_args.extend([generate_param_mapping[key], str(value)])
                else:
                    rest_args.extend(["-T", f"{key}={value}"])  # task parameter
        
        # Apply benchmark-specific overrides
        for key, value in benchmark_overrides.items():
            if key not in existing_t_args:
                if key in generate_param_mapping:
                    rest_args.extend([generate_param_mapping[key], str(value)])
                else:
                    rest_args.extend(["-T", f"{key}={value}"])  # task parameter
        
        # Add OpenAI-compatible API arguments (api_key, base_url)
        rest_args.extend(openai_compat_args)
    
    # Check WANDB environment variables
    if not _ensure_wandb_env():
        return 1
    
    # Build inspect eval command
    cmd = ["inspect", "eval", f"{horangi_py}@{benchmark}"] + rest_args
    
    # Execute (capture output to extract Weave Eval URL)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    weave_eval_url: str | None = None
    hook_noise_patterns = (
        r"^inspect_ai v",
        r"^- hooks enabled:",
        r"^\s*inspect_wandb/weave_evaluation_hooks:",
        r"^\s*inspect_wandb/wandb_models_hooks:",
        r"^\s*weave: Logged in as Weights & Biases user:",
        r"^Log: logs/",  # Hide local log file path
    )
    
    for line in process.stdout:
        # Extract Weave Eval URL
        m = re.search(r"🔗\s*Weave Eval:\s*(https?://\S+)", line)
        if m:
            weave_eval_url = m.group(1)
            continue  # Don't print URL line (print at the end)
        
        # Filter noise logs
        suppress = False
        for pat in hook_noise_patterns:
            if re.search(pat, line):
                suppress = True
                break
        
        if not suppress:
            print(line, end="", flush=True)
    
    process.wait()
    
    # Print Eval URL after evaluation completes
    if weave_eval_url:
        print()
        print(f"🔗 Weave Eval: {weave_eval_url}")
    
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
