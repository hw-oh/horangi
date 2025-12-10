"""
Inspect CLI에서 직접 실행 가능한 Task 정의 파일

사용법:
    # 단일 벤치마크 실행
    inspect eval eval_tasks.py@ko_hellaswag --model openai/gpt-4o
    inspect eval eval_tasks.py@ko_aime2025 --model openai/gpt-4o

    # 샘플 수 제한
    inspect eval eval_tasks.py@ko_aime2025 --model openai/gpt-4o -T limit=5

inspect-wandb가 설치되어 있으면 자동으로 WandB/Weave에 로깅됩니다.
"""

import sys
from pathlib import Path

# 프로젝트 소스를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============================================================================
# KoHellaSwag (Weave에서 데이터 로드)
# =============================================================================
from horangi.benchmarks.ko_hellaswag import (
    ko_hellaswag,
    ko_hellaswag_inherited,
)

# =============================================================================
# KoAIME2025 (로컬 JSONL에서 데이터 로드)
# =============================================================================
from horangi.benchmarks.aime2025 import ko_aime2025
