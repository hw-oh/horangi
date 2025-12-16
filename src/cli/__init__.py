#!/usr/bin/env python
"""
Horangi CLI - 한국어 LLM 벤치마크 평가 도구

사용법:
    uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5
    uv run horangi ko_hellaswag --config gpt-4o -T limit=5
    uv run horangi swebench_verified_official_80 --config claude-3-5-sonnet -T limit=1
    uv run horangi --list  # 사용 가능한 벤치마크 목록
    uv run horangi --list-models  # 사용 가능한 모델 설정 목록
    uv run horangi leaderboard --project <entity>/<project>  # 리더보드 생성
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def _ensure_wandb_env() -> bool:
    """
    WANDB_ENTITY와 WANDB_PROJECT 환경변수 확인 및 설정
    
    환경변수가 없으면 사용자에게 입력받아 설정합니다.
    
    Returns:
        True if 환경변수가 설정됨, False if 사용자가 취소
    """
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")
    
    if entity and project:
        return True
    
    print("⚠️  W&B 환경변수가 설정되지 않았습니다.")
    print()
    
    if not entity:
        try:
            entity = input("WANDB_ENTITY (팀 또는 사용자명): ").strip()
            if not entity:
                print("❌ WANDB_ENTITY가 필요합니다.")
                return False
            os.environ["WANDB_ENTITY"] = entity
        except (EOFError, KeyboardInterrupt):
            print("\n❌ 취소됨")
            return False
    
    if not project:
        try:
            project = input("WANDB_PROJECT (프로젝트명): ").strip()
            if not project:
                print("❌ WANDB_PROJECT가 필요합니다.")
                return False
            os.environ["WANDB_PROJECT"] = project
        except (EOFError, KeyboardInterrupt):
            print("\n❌ 취소됨")
            return False
    
    print()
    print(f"✅ 프로젝트: {entity}/{project}")
    print()
    
    return True


def _is_openai_compat_api(model_config: dict) -> bool:
    """
    OpenAI 호환 API인지 확인
    
    다음 조건 중 하나를 만족하면 OpenAI 호환 API:
    1. api_provider가 "openai"이고 base_url이 openai.com이 아닌 경우
    2. model_id가 "openai/"로 시작하고 base_url이 openai.com이 아닌 경우
    
    예: Solar, Grok, Together AI 등
    """
    api_provider = model_config.get("api_provider")
    model_id = model_config.get("model_id", "")
    base_url = model_config.get("base_url") or model_config.get("api_base")
    
    # api_provider가 openai이고 base_url이 openai.com이 아닌 경우
    if api_provider == "openai" and base_url:
        return "openai.com" not in base_url
    
    # 기존 방식: openai/ provider를 사용하면서 base_url이 openai.com이 아닌 경우
    if model_id.startswith("openai/") and base_url:
        return "openai.com" not in base_url
    
    return False


def _get_openai_compat_args(model_config: dict, verbose: bool = True) -> list[str]:
    """
    OpenAI 호환 API를 위한 CLI 인자 생성
    
    .env의 OPENAI_API_KEY가 아닌 모델 설정의 api_key_env에서 읽은 값을
    --model-args api_key=...로 직접 전달합니다.
    
    Returns:
        추가할 CLI 인자 리스트 (예: ["--model-args", "api_key=...", "--model-base-url", "..."])
    """
    extra_args = []
    
    if not _is_openai_compat_api(model_config):
        return extra_args
    
    # API 키: -M api_key=... 로 직접 전달 (.env 우회)
    api_key_env = model_config.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            extra_args.extend(["-M", f"api_key={api_key}"])
            if verbose:
                masked_key = api_key[:8] + "..." if len(api_key) > 8 else "***"
                print(f"🔑 {api_key_env} → -M api_key ({masked_key})")
        else:
            print(f"❌ 환경변수 {api_key_env}가 설정되지 않았습니다!")
            print(f"   다음 명령어로 설정하세요: export {api_key_env}=\"your-api-key\"")
    
    # Base URL: --model-base-url 로 전달
    base_url = model_config.get("base_url") or model_config.get("api_base")
    if base_url:
        extra_args.extend(["--model-base-url", base_url])
        if verbose:
            print(f"🌐 --model-base-url → {base_url}")
    
    return extra_args


def _handle_leaderboard_command(args: list[str]) -> int:
    """
    리더보드 생성 명령어 처리
    
    사용법:
        horangi leaderboard --project <entity>/<project>
        horangi leaderboard --project <entity>/<project> --name "My Leaderboard"
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Weave 리더보드 생성",
        prog="horangi leaderboard",
    )
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Weave 프로젝트 (예: my-team/my-project)",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="리더보드 이름 (기본: Korean LLM Leaderboard)",
    )
    parser.add_argument(
        "--description", "-d",
        default=None,
        help="리더보드 설명",
    )
    
    try:
        parsed = parser.parse_args(args)
    except SystemExit as e:
        return e.code if e.code else 0
    
    # 프로젝트에서 entity와 project 분리
    if "/" not in parsed.project:
        print("❌ 프로젝트 형식이 올바르지 않습니다. '<entity>/<project>' 형식을 사용하세요.")
        return 1
    
    entity, project = parsed.project.split("/", 1)
    
    print(f"🐯 Horangi - Weave 리더보드 생성")
    print(f"📁 프로젝트: {entity}/{project}")
    print()
    
    # src를 path에 추가
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
    
    # 프로젝트 루트 찾기 (src/cli/__init__.py -> 프로젝트 루트)
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    horangi_py = project_root / "horangi.py"
    
    # src를 path에 추가 (config_loader 등 사용 위해)
    sys.path.insert(0, str(src_path))
    
    # leaderboard: 리더보드 생성
    if args and args[0] == "leaderboard":
        return _handle_leaderboard_command(args[1:])
    
    # --list-models: 모델 설정 목록 출력
    if args and args[0] == "--list-models":
        print("🐯 Horangi - 사용 가능한 모델 설정")
        print()
        
        from core.config_loader import ConfigLoader
        config = ConfigLoader()
        models = config.list_models()
        
        if not models:
            print("  설정된 모델이 없습니다.")
            print(f"  configs/models/ 디렉토리에 YAML 파일을 추가하세요.")
        else:
            print("사용 가능한 모델 설정:")
            print()
            for model_name in sorted(models):
                if model_name.startswith("_"):  # 템플릿 파일 제외
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
                    print(f"  {'':25}   출시일: {release_date}")
                print()
        
        print("사용 예시:")
        print("  uv run horangi ko_hellaswag --config gpt-4o -T limit=5")
        return 0
    
    # --list 또는 -l 옵션: 벤치마크 목록 출력
    if not args or args[0] in ("--list", "-l", "--help", "-h"):
        print("🐯 Horangi - 한국어 LLM 벤치마크 평가 도구")
        print()
        print("사용법:")
        print("  uv run horangi <벤치마크> --model <모델> [옵션]")
        print("  uv run horangi <벤치마크> --config <설정파일> [옵션]")
        print()
        print("예시:")
        print("  uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5")
        print("  uv run horangi ko_hellaswag --config gpt-4o -T limit=5")
        print("  uv run horangi swebench_verified_official_80 --config claude-3-5-sonnet -T limit=1")
        print()
        print("모델 설정 목록:")
        print("  uv run horangi --list-models")
        print()
        print("리더보드 생성:")
        print("  uv run horangi leaderboard --project <entity>/<project>")
        print()
        
        # 벤치마크 목록 출력
        print("사용 가능한 벤치마크:")
        print()
        
        from benchmarks import list_benchmarks_with_descriptions
        
        # 카테고리별로 그룹화
        categories = {
            "일반": ["ko_hellaswag", "ko_aime2025", "ifeval_ko", "ko_balt_700"],
            "지식": ["haerae_bench_v1_rc", "haerae_bench_v1_wo_rc", "kmmlu", "kmmlu_pro", "squad_kor_v1", "ko_truthful_qa"],
            "추론": ["ko_moral", "ko_arc_agi", "ko_gsm8k"],
            "편향/안전": ["korean_hate_speech", "kobbq", "ko_hle"],
            "환각 (HalluLens)": ["ko_hallulens_wikiqa", "ko_hallulens_longwiki", "ko_hallulens_generated", "ko_hallulens_mixed", "ko_hallulens_nonexistent"],
            "Function Calling": ["bfcl"],
            "대화": ["mtbench_ko"],
            "코딩": ["swebench_verified_official_80"],
        }
        
        benchmarks_dict = dict(list_benchmarks_with_descriptions())
        
        for category, names in categories.items():
            print(f"  [{category}]")
            for name in names:
                desc = benchmarks_dict.get(name, "")
                print(f"    {name:<35} {desc}")
            print()
        
        print(f"총 {len(benchmarks_dict)}개 벤치마크")
        return 0
    
    # 첫 번째 인자가 벤치마크 이름
    benchmark = args[0]
    rest_args = list(args[1:])
    
    # --config 또는 -c 옵션 처리
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
                print("❌ --config 옵션에 모델 설정 이름이 필요합니다.")
                print("   예: --config gpt-4o")
                return 1
        new_args.append(arg)
        i += 1
    
    rest_args = new_args
    
    # 설정 파일에서 모델 정보 로드
    if config_name:
        from core.config_loader import ConfigLoader
        
        config = ConfigLoader()
        model_config = config.get_model(config_name)
        
        if not model_config:
            print(f"❌ 모델 설정을 찾을 수 없습니다: {config_name}")
            print(f"   사용 가능한 모델: {', '.join(config.list_models())}")
            return 1
        
        # OpenAI 호환 API 인자 생성 (Solar, Grok 등)
        # .env의 OPENAI_API_KEY 대신 모델 설정의 api_key_env 사용
        openai_compat_args = _get_openai_compat_args(model_config)
        
        # model_id와 api_provider 처리
        # model_id: 사용자가 보는 이름 (예: upstage/solar-pro2)
        # api_provider: 실제 API provider (예: openai - OpenAI 호환 API 사용 시)
        model_id = model_config.get("model_id", config_name)
        api_provider = model_config.get("api_provider")
        
        if api_provider:
            # api_provider가 지정된 경우: upstage/solar-pro2 → openai/solar-pro2
            model_name = model_id.split("/")[-1]  # 모델명만 추출
            inspect_model = f"{api_provider}/{model_name}"
        else:
            inspect_model = model_id
        
        # 이미 --model이 지정되어 있지 않으면 추가
        has_model = any(arg == "--model" for arg in rest_args)
        if not has_model:
            rest_args = ["--model", inspect_model] + rest_args
        
        # 벤치마크별 설정 적용
        benchmark_overrides = model_config.get("benchmarks", {}).get(benchmark, {})
        defaults = model_config.get("defaults", {})
        
        # 설정 적용 (-T 옵션으로 추가, 이미 지정된 것은 유지)
        existing_t_args = set()
        for j, arg in enumerate(rest_args):
            if arg == "-T" and j + 1 < len(rest_args):
                key = rest_args[j + 1].split("=")[0]
                existing_t_args.add(key)
        
        # defaults 적용
        for key, value in defaults.items():
            if key not in existing_t_args and key in ("temperature", "max_tokens"):
                rest_args.extend(["-T", f"{key}={value}"])
        
        # 벤치마크별 오버라이드 적용
        for key, value in benchmark_overrides.items():
            if key not in existing_t_args:
                rest_args.extend(["-T", f"{key}={value}"])
        
        # OpenAI 호환 API 인자 추가 (api_key, base_url)
        rest_args.extend(openai_compat_args)
    
    # WANDB 환경변수 확인
    if not _ensure_wandb_env():
        return 1
    
    # inspect eval 명령 구성
    cmd = ["inspect", "eval", f"{horangi_py}@{benchmark}"] + rest_args
    
    # 실행 (출력 캡처하여 Weave Eval URL 추출)
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
        r"^\s*weave: View Weave data at https://wandb.ai/",
    )
    
    for line in process.stdout:
        # Weave Eval URL 추출
        m = re.search(r"🔗\s*Weave Eval:\s*(https?://\S+)", line)
        if m:
            weave_eval_url = m.group(1)
            continue  # URL 라인은 출력하지 않음 (마지막에 출력)
        
        # 노이즈 로그 필터링
        suppress = False
        for pat in hook_noise_patterns:
            if re.search(pat, line):
                suppress = True
                break
        
        if not suppress:
            print(line, end="", flush=True)
    
    process.wait()
    
    # 평가 완료 후 Eval URL 출력
    if weave_eval_url:
        print()
        print(f"🔗 Weave Eval: {weave_eval_url}")
    
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
