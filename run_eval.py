#!/usr/bin/env python3
"""
모든 벤치마크를 5개 샘플씩 테스트하는 스크립트

사용법:
    uv run python run_all_benchmarks.py --model openai/gpt-4o-mini
    uv run python run_all_benchmarks.py --model openai/gpt-4o --limit 10
    uv run python run_all_benchmarks.py --model openai/gpt-4o --exclude haerae_bench_v1_rc,bfcl_text
"""

import argparse
import subprocess
import sys
from datetime import datetime

# 모든 벤치마크 목록
ALL_BENCHMARKS = [
    # "ko_hellaswag",
    # "ko_aime2025",
    # "ifeval_ko",
    # "ko_balt_700",
    # "haerae_bench_v1_rc",
    # "haerae_bench_v1_wo_rc",
    # "kmmlu",
    # "kmmlu_pro",
    # "squad_kor_v1",
    # "ko_truthful_qa",
    # "ko_moral",
    "ko_arc_agi",
    "ko_gsm8k",
    "korean_hate_speech",
    "kobbq",
    "ko_hle",
    "ko_hallulens_wikiqa",
    # "ko_hallulens_longwiki",
    # "ko_hallulens_generated",
    # "ko_hallulens_mixed",
    "ko_hallulens_nonexistent",
    "bfcl_extended",
    # "bfcl_text",
    "mtbench_ko",
    "swebench_verified_official_80",
]


def run_benchmark(benchmark: str, model: str, limit: int) -> tuple[str, bool, str]:
    """단일 벤치마크 실행"""
    cmd = [
        "uv", "run", "horangi",
        benchmark,
        "--model", model,
        "-T", f"limit={limit}",
    ]
    
    print(f"\n{'='*60}")
    print(f"🏃 Running: {benchmark}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=1800,  # 30분 타임아웃
        )
        success = result.returncode == 0
        return benchmark, success, "" if success else f"Exit code: {result.returncode}"
    except subprocess.TimeoutExpired:
        return benchmark, False, "Timeout (30m)"
    except Exception as e:
        return benchmark, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks with limited samples")
    parser.add_argument("--model", type=str, required=True, help="Model to use (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples per benchmark (default: 5)")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated list of benchmarks to exclude")
    parser.add_argument("--only", type=str, default="", help="Comma-separated list of benchmarks to run (exclusive)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    # 벤치마크 필터링
    if args.only:
        benchmarks = [b.strip() for b in args.only.split(",") if b.strip()]
        # 유효성 검사
        invalid = [b for b in benchmarks if b not in ALL_BENCHMARKS]
        if invalid:
            print(f"❌ Unknown benchmarks: {invalid}")
            print(f"   Available: {ALL_BENCHMARKS}")
            sys.exit(1)
    else:
        exclude_list = [b.strip() for b in args.exclude.split(",") if b.strip()]
        benchmarks = [b for b in ALL_BENCHMARKS if b not in exclude_list]
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n🔍 Dry run - commands that would be executed:")
        for benchmark in benchmarks:
            cmd = f"uv run horangi {benchmark} --model {args.model} -T limit={args.limit}"
            print(f"  {cmd}")
        return
    
    # 실행 결과 추적
    results = []
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] ", end="")
        name, success, error = run_benchmark(benchmark, args.model, args.limit)
        results.append((name, success, error))
    
    # 결과 요약
    print(f"\n\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"\n✅ Successful: {len(successful)} / {len(results)}")
    for name, _, _ in successful:
        print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} / {len(results)}")
        for name, _, error in failed:
            print(f"   - {name}: {error}")
    
    print(f"\n{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 실패가 있으면 exit code 1
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

