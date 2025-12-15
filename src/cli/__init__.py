#!/usr/bin/env python
"""
Horangi CLI - 한국어 LLM 벤치마크 평가 도구

사용법:
    uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5
    uv run horangi swebench_verified_official_80 --model openai/gpt-4o -T limit=1
    uv run horangi --list  # 사용 가능한 벤치마크 목록
"""

import subprocess
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    
    # 프로젝트 루트 찾기 (src/cli/__init__.py -> 프로젝트 루트)
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    horangi_py = project_root / "horangi.py"
    
    # --list 또는 -l 옵션: 벤치마크 목록 출력
    if not args or args[0] in ("--list", "-l", "--help", "-h"):
        print("🐯 Horangi - 한국어 LLM 벤치마크 평가 도구")
        print()
        print("사용법:")
        print("  uv run horangi <벤치마크> --model <모델> [옵션]")
        print()
        print("예시:")
        print("  uv run horangi ko_hellaswag --model openai/gpt-4o -T limit=5")
        print("  uv run horangi swebench_verified_official_80 --model openai/gpt-4o -T limit=1")
        print()
        
        # 벤치마크 목록 출력
        print("사용 가능한 벤치마크:")
        print()
        
        # src를 path에 추가하고 benchmarks import
        sys.path.insert(0, str(src_path))
        
        from benchmarks import list_benchmarks_with_descriptions
        
        # 카테고리별로 그룹화
        categories = {
            "일반": ["ko_hellaswag", "ko_aime2025", "ifeval_ko", "ko_balt_700"],
            "지식": ["haerae_bench_v1_rc", "haerae_bench_v1_wo_rc", "kmmlu", "kmmlu_pro", "squad_kor_v1", "ko_truthful_qa"],
            "추론": ["ko_moral", "ko_arc_agi", "ko_gsm8k"],
            "편향/안전": ["korean_hate_speech", "kobbq", "ko_hle"],
            "환각 (HalluLens)": ["ko_hallulens_wikiqa", "ko_hallulens_longwiki", "ko_hallulens_generated", "ko_hallulens_mixed", "ko_hallulens_nonexistent"],
            "Function Calling": ["bfcl_extended", "bfcl_text"],
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
    rest_args = args[1:]
    
    # inspect eval 명령 구성
    cmd = ["inspect", "eval", f"{horangi_py}@{benchmark}"] + rest_args
    # 실행
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
