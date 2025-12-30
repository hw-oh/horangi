import os

lgai_configs = [
    "EXAONE-4.0-1.2B",
    "EXAONE-4.0.1-32B",
    "EXAONE-4.0-32B"
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

vllm_configs = [
    # "gemma-3-270m-it",✅
    # "gemma-3-1b-it",✅
    # "gemma-3-4b-it",✅
    # "gemma-3-12b-it",✅
    # "gemma-3-27b-it",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
]
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default=None)
    args = parser.parse_args()

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
    elif args.provider == 'vllm':
        configs = vllm_configs
    else:
        configs = lgai_configs+openai_configs+anthropic_configs+xai_configs+google_configs+together_configs
    print("\n\n-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("Running evaluations for specified model configurations...")
    print("-----------------------------------------------------")
    print("-----------------------------------------------------\n\n")
    for config in configs:
        print(f"||||   uv run run_eval.py --config {config}   ||||\n\n")
        os.system(f"uv run run_eval.py --config {config}")