#!/usr/bin/env python3
"""
평가 결과로 리더보드 테이블 생성 CLI

이 모듈은 Inspect AI로 평가한 결과를 수집하여
W&B 리더보드 테이블로 변환합니다.

사용법:
    # 수동으로 결과 입력하여 리더보드 생성
    uv run python -m core.create_leaderboard_cli \
        --entity wandb-korea \
        --project korean-llm-eval \
        --model gpt-4o \
        --release-date 2024-05-13

    # Weave trace에서 자동으로 결과 수집
    uv run python -m core.create_leaderboard_cli \
        --entity wandb-korea \
        --project korean-llm-eval \
        --model gpt-4o \
        --from-weave

    # 여러 벤치마크 결과를 JSON으로 입력
    uv run python -m core.create_leaderboard_cli \
        --entity wandb-korea \
        --project korean-llm-eval \
        --model gpt-4o \
        --results '{"ko_hle": {"score": 0.42}, "kmmlu": {"score": 0.78}}'
"""

import argparse
import json
import sys

from core.leaderboard_table import (
    LeaderboardTableBuilder,
    BENCHMARK_CONFIG,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="평가 결과로 리더보드 테이블 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    # 수동으로 결과 입력
    python -m core.create_leaderboard_cli \\
        --entity wandb-korea \\
        --project korean-llm-eval \\
        --model gpt-4o \\
        --results '{"ko_hle": {"score": 0.42}, "kmmlu": {"score": 0.78}}'

    # Weave에서 자동 수집
    python -m core.create_leaderboard_cli \\
        --entity wandb-korea \\
        --project korean-llm-eval \\
        --model gpt-4o \\
        --from-weave

지원되는 벤치마크:
    - ko_hle, ko_aime2025, ko_gsm8k (추론)
    - kmmlu, kmmlu_pro, haerae_bench_v1_rc/wo_rc (지식)
    - ifeval_ko, ko_balt_700, ko_hellaswag (언어)
    - kobbq, ko_moral, korean_hate_speech (안전/편향)
    - ko_hallulens_* (환각 방지)
    - bfcl, swebench_verified_official_80 (도구/코딩)
    - mtbench_ko (대화)
"""
    )
    
    parser.add_argument(
        "--entity", "-e",
        required=True,
        help="W&B entity (팀 또는 사용자 이름)"
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="W&B 프로젝트 이름"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="평가 대상 모델 이름"
    )
    parser.add_argument(
        "--release-date",
        default="unknown",
        help="모델 출시일 (YYYY-MM-DD 형식)"
    )
    parser.add_argument(
        "--size-category",
        default="unknown",
        choices=["small", "medium", "large", "flagship", "unknown"],
        help="모델 크기 카테고리"
    )
    parser.add_argument(
        "--model-size",
        default="unknown",
        help="모델 파라미터 수 (예: 7B, 13B, 70B)"
    )
    parser.add_argument(
        "--results",
        type=str,
        help='벤치마크 결과 JSON 문자열 (예: \'{"ko_hle": {"score": 0.42}}\')'
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="벤치마크 결과가 담긴 JSON 파일 경로"
    )
    parser.add_argument(
        "--from-weave",
        action="store_true",
        help="Weave trace에서 결과 자동 수집"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="결과를 저장할 CSV 파일 경로"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="W&B에 로깅하지 않음 (로컬에서만 확인)"
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="지원되는 벤치마크 목록 출력"
    )
    
    return parser.parse_args()


def list_benchmarks():
    """지원되는 벤치마크 목록 출력"""
    print("\n📋 지원되는 벤치마크 목록:")
    print("=" * 60)
    
    # GLP 관련 벤치마크
    print("\n🎯 GLP (범용언어성능) 관련:")
    glp_benchmarks = []
    for name, config in BENCHMARK_CONFIG.items():
        mapper = config.get("mapper", {})
        for category in mapper.values():
            if category.startswith("GLP_"):
                glp_benchmarks.append((name, category))
                break
    
    for name, category in sorted(glp_benchmarks, key=lambda x: x[1]):
        print(f"  - {name:35} → {category}")
    
    # ALT 관련 벤치마크
    print("\n🛡️ ALT (가치정렬성능) 관련:")
    alt_benchmarks = []
    for name, config in BENCHMARK_CONFIG.items():
        mapper = config.get("mapper", {})
        for category in mapper.values():
            if category.startswith("ALT_"):
                alt_benchmarks.append((name, category))
                break
    
    for name, category in sorted(alt_benchmarks, key=lambda x: x[1]):
        print(f"  - {name:35} → {category}")
    
    print("\n" + "=" * 60)


def create_leaderboard_cli():
    """CLI 진입점"""
    args = parse_args()
    
    if args.list_benchmarks:
        list_benchmarks()
        return
    
    print(f"\n🏆 리더보드 테이블 생성기")
    print(f"{'=' * 60}")
    print(f"Entity:  {args.entity}")
    print(f"Project: {args.project}")
    print(f"Model:   {args.model}")
    print(f"{'=' * 60}")
    
    # 빌더 생성
    builder = LeaderboardTableBuilder(
        entity=args.entity,
        project=args.project,
        model_name=args.model,
        release_date=args.release_date,
        size_category=args.size_category,
        model_size=args.model_size,
    )
    
    # 결과 수집
    if args.from_weave:
        print("\n🔍 Weave trace에서 결과 수집 중...")
        builder.collect_from_weave_traces()
    
    if args.results:
        print("\n📥 JSON 문자열에서 결과 로드 중...")
        try:
            results = json.loads(args.results)
            for benchmark_name, scores in results.items():
                builder.add_benchmark_result(benchmark_name, scores)
                print(f"  ✓ {benchmark_name}: {scores}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 오류: {e}")
            sys.exit(1)
    
    if args.results_file:
        print(f"\n📂 파일에서 결과 로드 중: {args.results_file}")
        try:
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            for benchmark_name, scores in results.items():
                builder.add_benchmark_result(benchmark_name, scores)
                print(f"  ✓ {benchmark_name}: {scores}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"❌ 파일 로드 오류: {e}")
            sys.exit(1)
    
    # 결과 확인
    if not builder.benchmark_results:
        print("\n❌ 수집된 벤치마크 결과가 없습니다.")
        print("   --results, --results-file, 또는 --from-weave 옵션을 사용하세요.")
        sys.exit(1)
    
    print(f"\n📊 수집된 벤치마크: {len(builder.benchmark_results)}개")
    
    # 리더보드 생성
    if args.no_wandb:
        print("\n📋 리더보드 테이블 생성 중 (W&B 로깅 없음)...")
        df = builder.build_leaderboard_df()
    else:
        print("\n📋 리더보드 테이블 생성 및 W&B 로깅 중...")
        df = builder.build_and_log()
        builder.finish()
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 리더보드 테이블:")
    print("=" * 60)
    
    # 주요 점수 출력
    if 'FINAL_SCORE' in df.columns:
        print(f"\n🏆 FINAL_SCORE: {df['FINAL_SCORE'].iloc[0]:.4f}")
    if '범용언어성능(GLP)_AVG' in df.columns:
        print(f"   GLP 평균: {df['범용언어성능(GLP)_AVG'].iloc[0]:.4f}")
    if '가치정렬성능(ALT)_AVG' in df.columns:
        print(f"   ALT 평균: {df['가치정렬성능(ALT)_AVG'].iloc[0]:.4f}")
    
    print("\n📋 전체 테이블:")
    print(df.T.to_string())
    
    # CSV 저장
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n💾 결과가 {args.output}에 저장되었습니다.")
    
    print("\n✅ 완료!")
    
    return df


if __name__ == "__main__":
    create_leaderboard_cli()

