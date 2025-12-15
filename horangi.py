"""
한국어 LLM 벤치마크 평가 프레임워크

사용법:
    uv run inspect eval horangi@ko_hellaswag --model openai/gpt-4o -T limit=5
    uv run inspect eval horangi@swebench_verified_official_80 --model openai/gpt-4o -T limit=1

    # 옵션
    -T shuffle=true      # 데이터 셔플
    -T limit=10          # 샘플 수 제한
    -T split=train       # 데이터 분할 (weave 타입)
    -T use_korean_prompt=false  # 영어 프롬프트 사용

새 벤치마크 추가:
    1. src/benchmarks/ 폴더에 새 파일 생성
    2. benchmarks/__init__.py에 CONFIG import 추가
    3. 이 파일에 @task 함수 추가

inspect-wandb가 설치되어 있으면 자동으로 WandB/Weave에 로깅됩니다.
"""

import os
import sys
from pathlib import Path

# 로케일 설정 (inspect_ai 날짜 포맷 호환)
os.environ.setdefault("LC_TIME", "en_US.UTF-8")

# src 폴더를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inspect_ai import Task, task
from core import create_benchmark

# =============================================================================
# 벤치마크 Task 정의
# =============================================================================

@task
def ko_hellaswag(
    shuffle: bool = False,
    limit: int | None = None,
    split: str | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoHellaSwag"""
    return create_benchmark(
        name="ko_hellaswag",
        shuffle=shuffle,
        limit=limit,
        split=split,
        use_korean_prompt=use_korean_prompt,
    )


@task
def ko_aime2025(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoAIME2025"""
    return create_benchmark(
        name="ko_aime2025",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
    )


@task
def ko_balt_700(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoBALT-700"""
    return create_benchmark(
        name="ko_balt_700",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
    )

@task
def ko_balt_700_syntax(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoBALT-700-Syntax"""
    return create_benchmark(
        name="ko_balt_700_syntax",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
    )

@task
def ko_balt_700_semantic(
    shuffle: bool = False,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """KoBALT-700-Semantic"""
    return create_benchmark(
        name="ko_balt_700_semantic",
        shuffle=shuffle,
        limit=limit,
        use_korean_prompt=use_korean_prompt,
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
    """KoBBQ - 한국어 편향성 판단 벤치마크"""
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
    """KoHLE - 한국어 Humanity's Last Exam (상속 버전)"""
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
    """KoHLE Standalone - 한국어 HLE (독립 버전, 커스텀 scorer)"""
    return create_benchmark(
        name="ko_hle_standalone",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# HalluLens 벤치마크
# =============================================================================

@task
def ko_hallulens_wikiqa(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """KoHalluLens PreciseWikiQA - 위키피디아 기반 QA 환각 평가"""
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
    """KoHalluLens LongWiki - 긴 위키피디아 문서 기반 QA 환각 평가"""
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
    """KoHalluLens GeneratedEntity - 가상 엔티티 거부 평가"""
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
    """KoHalluLens MixedEntity - 실제/가상 혼합 엔티티 평가"""
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
    """KoHalluLens NonExistentEntities - 가상 엔티티 거부 평가 (Generated + Mixed 통합)"""
    return create_benchmark(
        name="ko_hallulens_nonexistent",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# BFCL 벤치마크
# =============================================================================

@task
def bfcl(
    shuffle: bool = False,
    limit: int | None = None,
    use_native_tools: bool = True,
) -> Task:
    """BFCL - Function Calling 벤치마크 (통합)
    
    모델의 tool calling 지원 여부에 따라 solver를 선택합니다.
    - use_native_tools=True (기본): Native Tool Calling (OpenAI, Claude, Gemini 등)
    - use_native_tools=False: Text-based (EXAONE, 일부 오픈소스 등)
    
    모델 설정 파일에서 자동으로 설정 가능:
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
    """[DEPRECATED] BFCL Extended - 대신 bfcl 벤치마크 사용 권장
    
    Tool calling 지원 모델용: OpenAI, Claude, Gemini 등
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
    """[DEPRECATED] BFCL Text - 대신 bfcl --use_native_tools=false 사용 권장
    
    Tool calling 미지원 모델용: EXAONE, 일부 오픈소스 등
    """
    return create_benchmark(
        name="bfcl_text",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# MT-Bench 벤치마크
# =============================================================================

@task
def mtbench_ko(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """MT-Bench 한국어 - 멀티턴 대화 평가
    
    8개 카테고리 (writing, roleplay, reasoning, math, coding, extraction, stem, humanities)
    2턴 대화 형식, LLM Judge가 1-10점 평가
    """
    return create_benchmark(
        name="mtbench_ko",
        shuffle=shuffle,
        limit=limit,
    )


# =============================================================================
# SWE-bench 벤치마크
# =============================================================================

@task
def swebench_verified_official_80(
    shuffle: bool = False,
    limit: int | None = None,
) -> Task:
    """SWE-bench Verified Official 80 - 소프트웨어 버그 수정 평가
    
    80개의 검증된 소프트웨어 버그 수정 과제
    서버 기반 채점 (Docker 컨테이너에서 테스트 실행)
    """
    return create_benchmark(
        name="swebench_verified_official_80",
        shuffle=shuffle,
        limit=limit,
    )

