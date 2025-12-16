#!/usr/bin/env python3

import argparse
import locale
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# inspect_evals의 날짜 파싱 문제 해결을 위해 영어 로케일 설정
try:
    locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, "C")
    except locale.Error:
        pass  # 로케일 설정 실패해도 계속 진행

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


def get_model_env(config_name: str) -> dict[str, str]:
    """
    모델 설정 파일에서 API 환경변수 생성
    
    configs/models/<config_name>.yaml 파일에서:
    - base_url → OPENAI_BASE_URL (또는 provider별 환경변수)
    - api_key_env → 해당 환경변수에서 API 키 읽기
    
    Returns:
        환경변수 딕셔너리
    """
    config = get_config()
    model_config = config.get_model(config_name)
    
    if not model_config:
        return {}
    
    env = {}
    
    # Provider 확인 (model_id 기준: openai/solar-pro2 → openai)
    model_id = model_config.get("model_id") or config_name
    provider = model_id.split("/")[0] if "/" in model_id else "openai"
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


def run_benchmark(
    benchmark: str, 
    config_name: str,
    limit: int | None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> tuple[str, bool, str, dict | None]:
    """
    단일 벤치마크 실행
    
    모델 설정 파일(configs/models/<model>.yaml)의 API 설정을 자동으로 적용합니다.
    
    Returns:
        (benchmark_name, success, error_message, scores)
    """
    cmd = ["uv", "run", "horangi", benchmark, "--config", config_name]
    
    # limit이 지정된 경우에만 추가 (null = 전체)
    if limit is not None:
        cmd.extend(["-T", f"limit={limit}"])
    
    # 모델 설정에서 환경변수 로드
    model_env = get_model_env(config_name)
    
    # 현재 환경변수와 병합 (모델 설정이 우선)
    env = os.environ.copy()
    env.update(model_env)
    
    # inspect_evals의 날짜 파싱 문제 해결을 위해 영어 로케일 설정
    env["LC_TIME"] = "en_US.UTF-8"

    # 각 벤치마크 subprocess(inspect eval)가 기록할 W&B/Weave 프로젝트 강제 지정
    # (지정하지 않으면 wandb의 기본 project(예: horangi-dev)로 기록될 수 있음)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    
    print(f"\n{'='*60}")
    print(f"🏃 Running: {benchmark}")
    print(f"   Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        # 실시간 출력을 위해 Popen 사용
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr를 stdout에 병합
            text=True,
            bufsize=1,  # 라인 버퍼링
            env=env,
        )
        
        # 실시간으로 출력하면서 결과 수집
        output_lines = []
        weave_eval_url: str | None = None
        hook_noise_patterns = (
            r"^inspect_ai v",
            r"^- hooks enabled:",
            r"^\s*inspect_wandb/weave_evaluation_hooks:",
            r"^\s*inspect_wandb/wandb_models_hooks:",
        )
        for line in process.stdout:
            # Weave Eval URL은 벤치마크 종료 후에 한 번만 보여주기 위해 캡처만 함
            m = re.search(r"🔗\s*Weave Eval:\s*(https?://\S+)", line)
            if m:
                weave_eval_url = m.group(1)
            
            # 불필요한 잡음 로그/중간 URL 라인 필터링
            suppress = False
            if m:
                suppress = True
            else:
                for pat in hook_noise_patterns:
                    if re.search(pat, line):
                        suppress = True
                        break
            
            if not suppress:
                print(line, end="", flush=True)  # 실시간 출력
            output_lines.append(line)
        
        process.wait(timeout=1800)  # 30분 타임아웃
        full_output = "".join(output_lines)
        
        success = process.returncode == 0
        
        # 벤치마크 종료 후 Weave Eval URL 출력
        if weave_eval_url:
            print(f"\n🔗 Weave Eval: {weave_eval_url}")
        
        # 점수 파싱 시도
        scores = None
        if success:
            scores = parse_scores_from_output(full_output, benchmark)
        
        return benchmark, success, "" if success else f"Exit code: {process.returncode}", scores
    
    except subprocess.TimeoutExpired:
        process.kill()
        return benchmark, False, "Timeout (30m)", None
    except Exception as e:
        if 'process' in locals():
            process.kill()
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
    
    Returns:
        {"score": 주요점수, "details": {메트릭명: 값, ...}}
    """
    all_metrics = {}
    
    # 모든 "이름  숫자" 패턴 파싱 (줄 시작, 밑줄/영문/숫자 이름)
    # stderr는 제외
    pattern = r"^([a-zA-Z][a-zA-Z0-9_]*)\s+([\d.-]+)\s*$"
    for match in re.finditer(pattern, output, re.MULTILINE):
        metric_name = match.group(1)
        # stderr, samples, tokens 등 메타 정보 제외
        if metric_name.lower() in ["stderr", "samples", "tokens", "total"]:
            continue
        try:
            all_metrics[metric_name] = float(match.group(2))
        except ValueError:
            pass
    
    if not all_metrics:
        return None
    
    # 주요 점수 선택
    main_score = None
    
    # IFEval: final_acc 또는 prompt_strict_acc 사용
    if benchmark == "ifeval_ko":
        main_score = all_metrics.get("final_acc") or all_metrics.get("prompt_strict_acc")
    
    # KoBBQ: kobbq_avg 사용
    elif benchmark == "kobbq":
        main_score = all_metrics.get("kobbq_avg")
    
    # HLE: hle_accuracy 사용
    elif benchmark == "ko_hle":
        main_score = all_metrics.get("hle_accuracy") or all_metrics.get("accuracy")
    
    # HalluLens: correct_rate 또는 refusal_rate 사용
    elif "hallulens" in benchmark:
        main_score = all_metrics.get("correct_rate") or all_metrics.get("refusal_rate")
    
    # MT-Bench: mean 사용 (10점 만점 → 0-1 스케일)
    elif benchmark == "mtbench_ko":
        if "mean" in all_metrics:
            main_score = all_metrics["mean"] / 10.0
    
    # BFCL: accuracy 사용
    elif benchmark == "bfcl":
        main_score = all_metrics.get("accuracy")
    
    # SQuAD: f1 > exact 우선순위
    elif benchmark == "squad_kor_v1":
        main_score = all_metrics.get("mean")  # f1.mean
    
    # 일반적인 메트릭 우선순위
    if main_score is None:
        for metric in ["accuracy", "mean", "macro_f1", "f1", "resolved"]:
            if metric in all_metrics:
                main_score = all_metrics[metric]
                if metric == "mean" and benchmark == "mtbench_ko":
                    main_score = main_score / 10.0  # mtbench 스케일
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
예제:
    # 기본 실행 (entity/project는 configs/base_config.yaml에서 로드)
    uv run python run_eval.py --config gpt-4o

    # 빠른 테스트 (가벼운 벤치마크만)
    uv run python run_eval.py --config gpt-4o --quick
    
    # 특정 벤치마크만 실행
    uv run python run_eval.py --config gpt-4o --only ko_hellaswag,kmmlu
"""
    )
    
    # 기본 옵션
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
    
    # base_config.yaml에서 W&B 설정 로드
    config = get_config()
    wandb_config = config.wandb
    
    entity = wandb_config.get("entity", "")
    project = wandb_config.get("project", "")
    
    if not entity or not project:
        print("❌ W&B 로깅을 위해 entity와 project가 필요합니다.")
        print("   configs/base_config.yaml의 wandb 섹션에 설정하세요.")
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
        benchmarks = ALL_BENCHMARKS
    
    # 모델 설정 로드 (configs/models/<name>.yaml)
    model_cfg = config.get_model(args.config)
    if not model_cfg:
        print(f"❌ 모델 설정을 찾을 수 없습니다: {args.config}")
        print("   configs/models/ 디렉토리에 YAML 파일이 있는지 확인하세요.")
        sys.exit(1)

    model_id = model_cfg.get("model_id") or args.config

    # 표시용 모델 이름 (openai/solar-pro2 → solar-pro2)
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    
    wandb_run = wandb.init(
        entity=entity,
        project=project,
        name=f"eval-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type="evaluation",
        config={
            "config": args.config,
            "model": model_id,
            "model_name": model_name,
            "limit": args.limit,
            "benchmarks": benchmarks,
        },
    )
    print(f"✅ W&B run 시작: {wandb_run.url}")
    
    
    print(f"\n🐯 Horangi Benchmark Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model: {model_id}")
    print(f"Limit: {args.limit} samples per benchmark")
    print(f"Benchmarks: {len(benchmarks)} / {len(ALL_BENCHMARKS)}")
    print(f"Leaderboard: {entity}/{project}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 실행 결과 추적
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
    
    # 카테고리별 상세 결과 테이블 출력
    print(f"\n{'='*60}")
    print(f"📋 Detailed Results by Category")
    print(f"{'='*60}")
    
    for benchmark_name, score_info in benchmark_scores.items():
        details = score_info.get("details", {})
        if len(details) > 1:  # 상세 결과가 있는 경우만
            print(f"\n📌 {benchmark_name}")
            print(f"   {'─'*40}")
            
            # 주요 메트릭과 카테고리별 메트릭 구분
            main_metrics = []
            category_metrics = []
            
            for metric, value in sorted(details.items()):
                if "_score" in metric or "_accuracy" in metric or "_rate" in metric or "_acc" in metric:
                    category_metrics.append((metric, value))
                else:
                    main_metrics.append((metric, value))
            
            # 주요 메트릭 출력
            for metric, value in main_metrics:
                print(f"   {metric:<30} {value:.4f}")
            
            # 카테고리별 메트릭 출력 (테이블 형식)
            if category_metrics:
                print(f"   {'─'*40}")
                for metric, value in category_metrics:
                    print(f"   {metric:<30} {value:.4f}")
    
    # Weave Leaderboard 생성 (성공한 벤치마크가 있으면)
    if benchmark_scores and entity and project:
        try:
            from core.weave_leaderboard import create_weave_leaderboard
            # 성공한 벤치마크 목록만 전달
            successful_benchmarks = list(benchmark_scores.keys())
            leaderboard_url = create_weave_leaderboard(
                entity=entity,
                project=project,
                benchmarks=successful_benchmarks,
            )
            if leaderboard_url:
                print(f"\n🏆 Leaderboard URL: {leaderboard_url}")
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
