"""
Korean LLM Benchmark Evaluation Framework

Usage:
    uv run inspect eval horangi@ko_hellaswag --model openai/gpt-4o -T limit=5
    uv run inspect eval horangi@swebench_verified_official_80 --model openai/gpt-4o -T limit=1

    # Options
    -T shuffle=true      # Shuffle data
    -T limit=10          # Limit sample count
    -T split=train       # Data split (weave type)

Adding new benchmarks:
    1. Create a new file in src/benchmarks/ folder
    2. Add CONFIG import to benchmarks/__init__.py
    3. Add @task function to this file

If inspect-wandb is installed, logging to WandB/Weave is automatic.
"""

import os
import sys
from pathlib import Path

# Set locale (for inspect_ai date format compatibility)
os.environ.setdefault("LC_TIME", "en_US.UTF-8")

# Add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_ai import Task, task
from core import create_benchmark

# =============================================================================
# Benchmark Task Definitions
# =============================================================================

@task
def ko_hellaswag(
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
) -> Task:
    """KoHellaSwag"""
    return create_benchmark(
        name="ko_hellaswag",
        shuffle=shuffle,
        limit=limit,
        split=split,
    )


@task
def ko_aime2025(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoAIME2025"""
    return create_benchmark(
        name="ko_aime2025",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_balt_700(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoBALT-700"""
    return create_benchmark(
        name="ko_balt_700",
        shuffle=shuffle,
        limit=limit,
    )

@task
def ko_balt_700_syntax(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoBALT-700-Syntax"""
    return create_benchmark(
        name="ko_balt_700_syntax",
        shuffle=shuffle,
        limit=limit,
    )

@task
def ko_balt_700_semantic(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoBALT-700-Semantic"""
    return create_benchmark(
        name="ko_balt_700_semantic",
        shuffle=shuffle,
        limit=limit,
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

@task
def haerae_bench_v1(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """HAERAE_BENCH_V1"""
    return create_benchmark(
        name="haerae_bench_v1",
        shuffle=shuffle,
        limit=limit,
    )

@task
def haerae_bench_v1_rc(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """HAERAE_BENCH_V1-RC"""
    return create_benchmark(
        name="haerae_bench_v1_rc",
        shuffle=shuffle,
        limit=limit,
    )

@task
def haerae_bench_v1_wo_rc(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """HAERAE_BENCH_V1-wo-RC"""
    return create_benchmark(
        name="haerae_bench_v1_wo_rc",
        shuffle=shuffle,
        limit=limit,
    )

@task
def kmmlu(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KMMLU"""
    return create_benchmark(
        name="kmmlu",
        shuffle=shuffle,
        limit=limit,
    )

@task
def kmmlu_pro(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KMMLU-Pro"""
    return create_benchmark(
        name="kmmlu_pro",
        shuffle=shuffle,
        limit=limit,
    )

@task
def squad_kor_v1(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """Squad-Kor-V1"""
    return create_benchmark(
        name="squad_kor_v1",
        shuffle=shuffle,
        limit=limit,
    )

@task
def ko_truthful_qa(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoTruthfulQA"""
    return create_benchmark(
        name="ko_truthful_qa",
        shuffle=shuffle,
        limit=limit,
    )

@task
def ko_moral(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoMoral"""
    return create_benchmark(
        name="ko_moral",
        shuffle=shuffle,
        limit=limit,
    )

@task
def ko_arc_agi(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """Ko-ARC-AGI"""
    return create_benchmark(
        name="ko_arc_agi",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_gsm8k(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoGSM8K"""
    return create_benchmark(
        name="ko_gsm8k",
        shuffle=shuffle,
        limit=limit,
    )


@task
def korean_hate_speech(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """Korean Hate Speech Detection"""
    return create_benchmark(
        name="korean_hate_speech",
        shuffle=shuffle,
        limit=limit,
    )


@task
def kobbq(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoBBQ - Korean Bias Benchmark for QA"""
    return create_benchmark(
        name="kobbq",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hle(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHLE - Korean Humanity's Last Exam (inherited version)"""
    return create_benchmark(
        name="ko_hle",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hle_standalone(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHLE Standalone - Korean HLE (standalone version, custom scorer)"""
    return create_benchmark(
        name="ko_hle_standalone",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# HalluLens Benchmarks
# =============================================================================

@task
def ko_hallulens_wikiqa(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens PreciseWikiQA - Wikipedia-based QA hallucination evaluation"""
    return create_benchmark(
        name="ko_hallulens_wikiqa",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hallulens_longwiki(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens LongWiki - Long Wikipedia document QA hallucination evaluation"""
    return create_benchmark(
        name="ko_hallulens_longwiki",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hallulens_generated(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens GeneratedEntity - Non-existent entity refusal evaluation"""
    return create_benchmark(
        name="ko_hallulens_generated",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hallulens_mixed(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens MixedEntity - Real/fictional mixed entity evaluation"""
    return create_benchmark(
        name="ko_hallulens_mixed",
        shuffle=shuffle,
        limit=limit,
    )


@task
def ko_hallulens_nonexistent(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens NonExistentEntities - Non-existent entity refusal evaluation (Generated + Mixed combined)"""
    return create_benchmark(
        name="ko_hallulens_nonexistent",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# BFCL Benchmarks
# =============================================================================

@task
def bfcl(
    shuffle: bool = False,
    limit: int | None = None,
    use_native_tools: bool = True,
) -> Task:
    """BFCL - Function Calling Benchmark (unified)
    
    Selects solver based on model's tool calling support.
    - use_native_tools=True (default): Native Tool Calling (OpenAI, Claude, Gemini, etc.)
    - use_native_tools=False: Text-based (EXAONE, some open-source models, etc.)
    
    Can be set automatically via model config file:
        benchmarks:
          bfcl:
            use_native_tools: false
    """
    return create_benchmark(
        name="bfcl",
        shuffle=shuffle,
        limit=limit,
        use_native_tools=use_native_tools,
    )


@task
def bfcl_extended(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """[DEPRECATED] BFCL Extended - Use bfcl benchmark instead
    
    For models supporting tool calling: OpenAI, Claude, Gemini, etc.
    """
    return create_benchmark(
        name="bfcl_extended",
        shuffle=shuffle,
        limit=limit,
    )


@task
def bfcl_text(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """[DEPRECATED] BFCL Text - Use bfcl --use_native_tools=false instead
    
    For models without tool calling support: EXAONE, some open-source models, etc.
    """
    return create_benchmark(
        name="bfcl_text",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# MT-Bench Benchmark
# =============================================================================

@task
def ko_mtbench(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """MT-Bench Korean - Multi-turn conversation evaluation
    
    8 categories (writing, roleplay, reasoning, math, coding, extraction, stem, humanities)
    2-turn conversation format, LLM Judge scores 1-10
    """
    return create_benchmark(
        name="ko_mtbench",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# SWE-bench Benchmark
# =============================================================================

@task
def swebench_verified_official_80(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """SWE-bench Verified Official 80 - Software bug fix evaluation
    
    80 verified software bug fix tasks
    Server-based scoring (runs tests in Docker container)
    """
    return create_benchmark(
        name="swebench_verified_official_80",
        shuffle=shuffle,
        limit=limit,
    )
