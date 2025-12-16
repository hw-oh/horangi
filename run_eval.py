#!/usr/bin/env python3
"""
모든 벤치마크 실행 및 리더보드 생성 스크립트

사용법:
    # 기본 실행 (5샘플씩) - configs/models/gpt-4o.yaml 설정 자동 적용
    uv run python run_eval.py --model openai/gpt-4o-mini

    # 더 많은 샘플로 실행
    uv run python run_eval.py --model openai/gpt-4o --limit 10

    # 특정 벤치마크 제외
    uv run python run_eval.py --model openai/gpt-4o --exclude haerae_bench_v1_rc,bfcl_text

    # 특정 벤치마크만 실행
    uv run python run_eval.py --model openai/gpt-4o --only ko_hle,kmmlu,kobbq

    # W&B 리더보드 자동 생성
    uv run python run_eval.py --model openai/gpt-4o \
        --entity wandb-korea \
        --project korean-llm-eval \
        --create-leaderboard

    # 리더보드 생성 + 모델 메타데이터 (설정 파일에서 자동 로드)
    uv run python run_eval.py --model openai/gpt-4o \
        --entity wandb-korea \
        --project korean-llm-eval \
        --create-leaderboard

Note:
    모델 설정은 configs/models/<model_name>.yaml 파일에서 자동으로 로드됩니다.
    - base_url: API 엔드포인트 (OPENAI_BASE_URL 등으로 설정됨)
    - api_key_env: API 키 환경변수 이름
    - metadata: release_date, size_category 등 리더보드용 메타데이터
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

import wandb
from core.config_loader import get_config

# 모든 벤치마크 목록 (활성화된 것만)
ALL_BENCHMARKS = [
    "ko_hellaswag",
    "ko_aime2025",
    "ifeval_ko",
    "ko_balt_700_syntax",
    "ko_balt_700_semantic",
    "haerae_bench_v1_rc",
    "haerae_bench_v1_wo_rc",
    "kmmlu",
    "kmmlu_pro",
    "squad_kor_v1",
    "ko_truthful_qa",
    "ko_moral",
    "ko_arc_agi",
    "ko_gsm8k",
    "korean_hate_speech",
    "kobbq",
    "ko_hle",
    "ko_hallulens_wikiqa",
    "ko_hallulens_longwiki",
    "ko_hallulens_nonexistent",
    "bfcl",
    "mtbench_ko",
    "swebench_verified_official_80",
]

# 빠른 테스트용 벤치마크 (가벼운 것들만)
QUICK_BENCHMARKS = [
    "ko_hellaswag",
    "kmmlu",
    "kobbq",
    "korean_hate_speech",
    "ifeval_ko",
    "ko_moral",
]


def get_model_env(model: str) -> dict[str, str]:
    """
    모델 설정 파일에서 API 환경변수 생성
    
    configs/models/<model_name>.yaml 파일에서:
    - base_url → OPENAI_BASE_URL (또는 provider별 환경변수)
    - api_key_env → 해당 환경변수에서 API 키 읽기
    
    Returns:
        환경변수 딕셔너리
    """
    config = get_config()
    model_config = config.get_model(model)
    
    if not model_config:
        return {}
    
    env = {}
    
    # Provider 확인 (openai/gpt-4o → openai)
    provider = model.split("/")[0] if "/" in model else "openai"
    provider_upper = provider.upper()
    
    # Base URL 설정
    base_url = model_config.get("base_url") or model_config.get("api_base")
    if base_url:
        # OpenAI 호환 API는 OPENAI_BASE_URL 사용
        if provider in ["openai", "together", "groq", "fireworks"]:
            env["OPENAI_BASE_URL"] = base_url
        else:
            env[f"{provider_upper}_BASE_URL"] = base_url
    
    # API 키 설정
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            # OpenAI 호환 API
            if provider in ["openai", "together", "groq", "fireworks"]:
                env["OPENAI_API_KEY"] = api_key
            else:
                env[f"{provider_upper}_API_KEY"] = api_key
    
    return env


def get_model_metadata(model: str) -> dict:
    """
    모델 설정 파일에서 메타데이터 로드
    
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


def run_benchmark(benchmark: str, model: str, limit: int) -> tuple[str, bool, str, dict | None]:
    """
    단일 벤치마크 실행
    
    모델 설정 파일(configs/models/<model>.yaml)의 API 설정을 자동으로 적용합니다.
    
    Returns:
        (benchmark_name, success, error_message, scores)
    """
    cmd = [
        "uv", "run", "horangi",
        benchmark,
        "--model", model,
        "-T", f"limit={limit}",
    ]
    
    # 모델 설정에서 환경변수 로드
    model_env = get_model_env(model)
    
    # 현재 환경변수와 병합 (모델 설정이 우선)
    env = os.environ.copy()
    env.update(model_env)
    
    print(f"\n{'='*60}")
    print(f"🏃 Running: {benchmark}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30분 타임아웃
            env=env,  # 모델 설정이 적용된 환경변수 사용
        )
        
        success = result.returncode == 0
        
        # stdout 출력
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # 점수 파싱 시도
        scores = None
        if success:
            scores = parse_scores_from_output(result.stdout + result.stderr, benchmark)
        
        return benchmark, success, "" if success else f"Exit code: {result.returncode}", scores
    
    except subprocess.TimeoutExpired:
        return benchmark, False, "Timeout (30m)", None
    except Exception as e:
        return benchmark, False, str(e), None


def parse_scores_from_output(output: str, benchmark: str) -> dict | None:
    """
    Inspect AI 출력에서 점수 파싱
    
    Inspect AI 출력 형식 예시:
        accuracy  0.600
        stderr    0.245
        
        또는
        
        mean    0.640
        writing_score  0.640
    """
    scores = {}
    
    # Inspect AI 출력 형식: "metric_name  value" (공백으로 구분, 줄 시작)
    # 각 메트릭별 패턴 (이름, 정규식)
    metric_patterns = [
        ("accuracy", r"^accuracy\s+([\d.]+)", False),
        ("mean", r"^mean\s+([\d.]+)", False),  # mtbench
        ("macro_f1", r"macro_f1\s+([\d.]+)", False),
        ("f1", r"^f1\s+([\d.]+)", False),
        ("resolved", r"resolved\s+([\d.]+)", False),  # swebench
        ("refusal_rate", r"refusal_rate\s+([\d.]+)", False),  # hallulens
        ("correct_rate", r"correct_rate\s+([\d.]+)", False),  # hallulens
        ("kobbq_avg", r"kobbq_avg\s+([\d.]+)", False),  # kobbq
        ("final_acc", r"final_acc\s+([\d.]+)", False),  # ifeval
        ("prompt_strict_acc", r"prompt_strict_acc\s+([\d.]+)", False),  # ifeval
        ("hle_accuracy", r"hle_accuracy\s+([\d.]+)", False),  # hle
    ]
    
    for metric_name, pattern, _ in metric_patterns:
        match = re.search(pattern, output, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                scores[metric_name] = float(match.group(1))
            except ValueError:
                pass
    
    # 벤치마크별 주요 점수 선택
    # IFEval: final_acc 또는 prompt_strict_acc 사용
    if benchmark == "ifeval_ko":
        if "final_acc" in scores:
            return {"score": scores["final_acc"]}
        elif "prompt_strict_acc" in scores:
            return {"score": scores["prompt_strict_acc"]}
    
    # KoBBQ: kobbq_avg 사용
    if benchmark == "kobbq" and "kobbq_avg" in scores:
        return {"score": scores["kobbq_avg"]}
    
    # HLE: hle_accuracy 사용
    if benchmark == "ko_hle" and "hle_accuracy" in scores:
        return {"score": scores["hle_accuracy"]}
    
    # HalluLens: correct_rate 또는 refusal_rate 사용
    if "hallulens" in benchmark:
        if "correct_rate" in scores:
            return {"score": scores["correct_rate"]}
        elif "refusal_rate" in scores:
            return {"score": scores["refusal_rate"]}
    
    # MT-Bench: mean 사용 (10점 만점 → 0-1 스케일)
    if benchmark == "mtbench_ko" and "mean" in scores:
        return {"score": scores["mean"] / 10.0}
    
    # 일반적인 메트릭 우선순위
    if "accuracy" in scores:
        return {"score": scores["accuracy"]}
    elif "mean" in scores:
        return {"score": scores["mean"] / 10.0}
    elif "macro_f1" in scores:
        return {"score": scores["macro_f1"]}
    elif "f1" in scores:
        return {"score": scores["f1"]}
    elif "resolved" in scores:
        return {"score": scores["resolved"]}
    
    return None


def create_leaderboard(
    model: str,
    benchmark_scores: dict[str, dict],
    entity: str,
    project: str,
    release_date: str = "unknown",
    size_category: str = "unknown",
    model_size: str = "unknown",
    wandb_run=None,
    output_csv: str | None = None,
):
    """
    벤치마크 결과로 리더보드 테이블 생성
    
    Args:
        wandb_run: 기존 W&B run 객체 (있으면 해당 run에 로깅, 없으면 새 run 생성)
    """
    from core.leaderboard_table import LeaderboardTableBuilder
    
    print(f"\n{'='*60}")
    print(f"🏆 리더보드 테이블 생성")
    print(f"{'='*60}")
    
    # 모델 이름에서 provider 제거 (openai/gpt-4o → gpt-4o)
    model_name = model.split("/")[-1] if "/" in model else model
    
    builder = LeaderboardTableBuilder(
        entity=entity,
        project=project,
        model_name=model_name,
        release_date=release_date,
        size_category=size_category,
        model_size=model_size,
    )
    
    # 벤치마크 결과 추가
    for benchmark_name, scores in benchmark_scores.items():
        if scores:
            builder.add_benchmark_result(benchmark_name, scores)
            print(f"  ✓ {benchmark_name}: {scores}")
    
    if not builder.benchmark_results:
        print("❌ 수집된 벤치마크 결과가 없습니다.")
        return None
    
    print(f"\n📊 수집된 벤치마크: {len(builder.benchmark_results)}개")
    
    # 리더보드 DataFrame 생성
    print("\n📋 리더보드 테이블 생성 중...")
    try:
        df = builder.build_leaderboard_df()
        glp_radar, glp_detail, alt_radar, alt_detail = builder.build_radar_tables(df)
    except Exception as e:
        print(f"⚠️ 테이블 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # W&B에 로깅 (기존 run 사용)
    if wandb_run is not None:
        print("📤 W&B run에 리더보드 테이블 로깅 중...")
        try:
            # 리더보드 테이블
            leaderboard_table = wandb.Table(dataframe=df)
            wandb_run.log({"leaderboard_table": leaderboard_table})
            
            # 레이더 테이블
            wandb_run.log({
                "glp_radar_table": wandb.Table(dataframe=glp_radar),
                "glp_detail_radar_table": wandb.Table(dataframe=glp_detail),
                "alt_radar_table": wandb.Table(dataframe=alt_radar),
                "alt_detail_radar_table": wandb.Table(dataframe=alt_detail),
            })
            
            # Summary에 주요 점수 저장
            if 'FINAL_SCORE' in df.columns and len(df) > 0:
                score = df['FINAL_SCORE'].iloc[0]
                if score == score:  # not NaN
                    wandb_run.summary["FINAL_SCORE"] = score
            if '범용언어성능(GLP)_AVG' in df.columns and len(df) > 0:
                score = df['범용언어성능(GLP)_AVG'].iloc[0]
                if score == score:
                    wandb_run.summary["GLP_AVG"] = score
            if '가치정렬성능(ALT)_AVG' in df.columns and len(df) > 0:
                score = df['가치정렬성능(ALT)_AVG'].iloc[0]
                if score == score:
                    wandb_run.summary["ALT_AVG"] = score
            
            print("✅ W&B 로깅 완료!")
        except Exception as e:
            print(f"⚠️ W&B 로깅 실패: {e}")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 리더보드 테이블:")
    print("=" * 60)
    
    # 주요 점수 출력
    if 'FINAL_SCORE' in df.columns and len(df) > 0:
        score = df['FINAL_SCORE'].iloc[0]
        if not (score != score):  # NaN check
            print(f"\n🏆 FINAL_SCORE: {score:.4f}")
    if '범용언어성능(GLP)_AVG' in df.columns and len(df) > 0:
        score = df['범용언어성능(GLP)_AVG'].iloc[0]
        if not (score != score):
            print(f"   GLP 평균: {score:.4f}")
    if '가치정렬성능(ALT)_AVG' in df.columns and len(df) > 0:
        score = df['가치정렬성능(ALT)_AVG'].iloc[0]
        if not (score != score):
            print(f"   ALT 평균: {score:.4f}")
    
    print("\n📋 전체 테이블:")
    print(df.T.to_string())
    
    # CSV 저장
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 결과가 {output_csv}에 저장되었습니다.")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks and create leaderboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
    # 기본 실행
    uv run python run_eval.py --model openai/gpt-4o-mini

    # 전체 벤치마크 + 리더보드 생성
    uv run python run_eval.py --model openai/gpt-4o \\
        --entity wandb-korea \\
        --project korean-llm-eval \\
        --create-leaderboard \\
        --release-date 2024-05-13

    # 빠른 테스트 (가벼운 벤치마크만)
    uv run python run_eval.py --model openai/gpt-4o-mini --quick
"""
    )
    
    # 기본 옵션
    parser.add_argument("--model", type=str, required=True, 
                        help="Model to use (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--limit", type=int, default=5, 
                        help="Number of samples per benchmark (default: 5)")
    parser.add_argument("--exclude", type=str, default="", 
                        help="Comma-separated list of benchmarks to exclude")
    parser.add_argument("--only", type=str, default="", 
                        help="Comma-separated list of benchmarks to run (exclusive)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only quick/light benchmarks")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without running")
    
    # 리더보드 옵션
    parser.add_argument("--create-leaderboard", action="store_true",
                        help="Create leaderboard table after running benchmarks")
    parser.add_argument("--entity", "-e", type=str, default="",
                        help="W&B entity (required for leaderboard)")
    parser.add_argument("--project", "-p", type=str, default="",
                        help="W&B project (required for leaderboard)")
    parser.add_argument("--release-date", type=str, default="unknown",
                        help="Model release date (YYYY-MM-DD)")
    parser.add_argument("--size-category", type=str, default="unknown",
                        choices=["small", "medium", "large", "flagship", "unknown"],
                        help="Model size category")
    parser.add_argument("--model-size", type=str, default="unknown",
                        help="Model parameter count (e.g., 7B, 13B, 70B)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Don't log to W&B (local only)")
    parser.add_argument("--output-csv", type=str,
                        help="Save leaderboard to CSV file")
    
    args = parser.parse_args()
    
    # 리더보드 생성 시 entity/project 필수
    if args.create_leaderboard and not args.no_wandb:
        if not args.entity or not args.project:
            print("❌ 리더보드 생성 시 --entity와 --project가 필요합니다.")
            print("   또는 --no-wandb 옵션으로 W&B 로깅 없이 로컬에서만 생성할 수 있습니다.")
            sys.exit(1)
    
    # 벤치마크 필터링
    if args.quick:
        benchmarks = QUICK_BENCHMARKS
    elif args.only:
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
    
    # 모델 이름에서 provider 제거 (openai/gpt-4o → gpt-4o)
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    if args.create_leaderboard:
        print(f"Leaderboard: {args.entity}/{args.project}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n🔍 Dry run - commands that would be executed:")
        for benchmark in benchmarks:
            cmd = f"uv run horangi {benchmark} --model {args.model} -T limit={args.limit}"
            print(f"  {cmd}")
        return
    
    # W&B run 초기화 (리더보드 생성 시)
    wandb_run = None
    if args.create_leaderboard and not args.no_wandb:
        print(f"\n📊 W&B run 초기화 중...")
        try:
            wandb_run = wandb.init(
                entity=args.entity,
                project=args.project,
                name=f"eval-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                job_type="evaluation",
                config={
                    "model": args.model,
                    "model_name": model_name,
                    "limit": args.limit,
                    "benchmarks": benchmarks,
                },
            )
            print(f"✅ W&B run 시작: {wandb_run.url}")
        except Exception as e:
            print(f"⚠️ W&B 초기화 실패: {e}")
            print("   로컬에서만 실행합니다.")
            wandb_run = None
    
    # 실행 결과 추적
    results = []
    benchmark_scores = {}
    
    for i, benchmark in enumerate(benchmarks, 1):
        print(f"\n[{i}/{len(benchmarks)}] ", end="")
        name, success, error, scores = run_benchmark(benchmark, args.model, args.limit)
        results.append((name, success, error))
        
        if scores:
            benchmark_scores[name] = scores
            
            # W&B에 개별 벤치마크 점수 로깅
            if wandb_run is not None:
                try:
                    wandb_run.log({
                        f"benchmark/{name}": scores.get("score", 0),
                    })
                except Exception:
                    pass
    
    # 결과 요약
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
    
    # W&B summary에 성공/실패 수 기록
    if wandb_run is not None:
        wandb_run.summary["successful_benchmarks"] = len(successful)
        wandb_run.summary["failed_benchmarks"] = len(failed)
        wandb_run.summary["total_benchmarks"] = len(results)
    
    # 리더보드 생성 (성공한 벤치마크가 있으면)
    if args.create_leaderboard and benchmark_scores:
        # 모델 설정에서 메타데이터 로드 (CLI 인자가 없으면 설정 파일 사용)
        model_metadata = get_model_metadata(args.model)
        
        release_date = args.release_date if args.release_date != "unknown" else model_metadata.get("release_date", "unknown")
        size_category = args.size_category if args.size_category != "unknown" else model_metadata.get("size_category", "unknown")
        model_size = args.model_size if args.model_size != "unknown" else model_metadata.get("model_size", "unknown")
        
        # 1. W&B Models 테이블 리더보드 생성
        create_leaderboard(
            model=args.model,
            benchmark_scores=benchmark_scores,
            entity=args.entity or "local",
            project=args.project or "benchmark-results",
            release_date=release_date,
            size_category=size_category,
            model_size=model_size,
            wandb_run=wandb_run,
            output_csv=args.output_csv,
        )
        
        # 2. Weave Leaderboard 생성 (별도 기능)
        if args.entity and args.project:
            try:
                from core.weave_leaderboard import create_weave_leaderboard
                # 성공한 벤치마크 목록만 전달
                successful_benchmarks = list(benchmark_scores.keys())
                create_weave_leaderboard(
                    entity=args.entity,
                    project=args.project,
                    benchmarks=successful_benchmarks,
                )
            except Exception as e:
                print(f"⚠️ Weave Leaderboard 생성 실패: {e}")
    
    # W&B run 종료
    if wandb_run is not None:
        print(f"\n📊 W&B run 종료 중...")
        wandb_run.finish()
        print(f"✅ W&B run 완료!")
    
    print(f"\n{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 실패가 있어도 리더보드가 생성되었으면 exit code 0
    # (일부 벤치마크 실패해도 결과는 저장됨)
    sys.exit(0)


if __name__ == "__main__":
    main()
