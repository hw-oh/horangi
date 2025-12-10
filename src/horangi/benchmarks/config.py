"""
벤치마크 설정 파일

새 벤치마크를 추가하려면 BENCHMARKS 딕셔너리에 설정을 추가하세요.

설정 필드:
- base: inspect_evals 벤치마크 이름 (solver/scorer 재사용) - 선택
- data_type: "weave" 또는 "jsonl"
- data_source: Weave ref 또는 로컬 파일 경로 (data/ 기준)
- split: 데이터 분할 (weave 타입일 때만) - 선택
- field_mapping: 데이터 필드 → Sample 필드 매핑
- target_transform: target 변환 함수 이름 - 선택
- system_message: 한국어 시스템 메시지 - 선택

Solver/Scorer 설정 (우선순위: 직접 지정 > base > 기본값):
- solver: "multiple_choice" | "generate"
- scorer: "choice" | "match" | "match_numeric"
- custom_scorer: scorers.py에 정의된 커스텀 scorer 이름
"""

from typing import Literal, Callable, Any

# =============================================================================
# 벤치마크 설정
# =============================================================================

BENCHMARKS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # KoHellaSwag - 문장 완성 상식 추론
    # =========================================================================
    "ko_hellaswag": {
        "base": "hellaswag",
        "data_type": "weave",
        "data_source": "weave:///wandb-korea/evaluation-job/object/KoHellaSwag:PY229AMRxLFoCLqKsaEguY4jVCuvyoMnQ5wJ1wkrXfU",
        "split": "train",
        "field_mapping": {
            "id": ["id", "ind"],  # 첫 번째 있는 것 사용
            "input": "ctx",
            "target": "label",
            "choices": "endings",
        },
        "target_transform": "label_to_letter",  # 0-3 → A-D
        "system_message": "이야기의 가장 자연스러운 다음 문장을 선택하세요.",
        "metadata": {
            "task_type": "commonsense_reasoning",
        },
    },
    
    # =========================================================================
    # KoAIME2025 - 수학 경시대회 문제
    # =========================================================================
    "ko_aime2025": {
        "base": "aime2025",
        "data_type": "jsonl",
        "data_source": "ko_aime2025.jsonl",  # data/ 폴더 기준 상대 경로
        "field_mapping": {
            "id": "id",
            "input": "problem",
            "target": "answer",
        },
        "target_transform": "to_string",  # 정수 → 문자열
        "metadata": {
            "task_type": "math_reasoning",
        },
    },
    
    # =========================================================================
    # KoBALT-700 - 원본 없이 직접 정의
    # =========================================================================
    "ko_balt_700": {
        "data_type": "weave",
        "data_source": "weave:///wandb-korea/evaluation-job/object/KoBALT-700:4g1U9ysNXVYSgiHu5u1tKD8wFyhqjcHwm82m70Idk5g",
        "field_mapping": {
            "id": "id",
            "input": "question",
            "target": "answer",
            "choices": "options",
        },
        "solver": "multiple_choice",  # 명시적 지정
        "scorer": "choice",           # 명시적 지정
        "target_transform": "identity",
        "system_message": "주어진 질문에 가장 적절한 답을 선택하세요.",
    },

    # =========================================================================
    # IFEval-Ko
    # =========================================================================
    "ifeval_ko": {
        "base": "ifeval",
        "data_type": "weave",
        "data_source": "weave:///wandb-korea/evaluation-job/object/IFEval-Ko:SGtm8r2dBuXUnkS402O7vYwzrBGrzYCfrLu7WS2u2No",
    },
    
    # =========================================================================
    # 추가 벤치마크 예시 (주석 처리)
    # =========================================================================
    # "ko_bbq": {
    #     "base": "bbq",
    #     "data_type": "weave",
    #     "data_source": "weave:///...",
    #     "weave_project": "wandb-korea/evaluation-job",
    #     "field_mapping": {...},
    #     "custom_scorer": "ko_bbq_scorer",  # scorers.py에 정의 필요
    # },
}


# =============================================================================
# Target 변환 함수들
# =============================================================================

def label_to_letter(label: Any) -> str:
    """숫자 라벨(0-3)을 문자(A-D)로 변환"""
    return chr(ord("A") + int(label))


def to_string(value: Any) -> str:
    """값을 문자열로 변환"""
    return str(value)


def identity(value: Any) -> Any:
    """변환 없이 그대로 반환"""
    return value


TARGET_TRANSFORMS: dict[str, Callable] = {
    "label_to_letter": label_to_letter,
    "to_string": to_string,
    "identity": identity,
}


# =============================================================================
# 유틸리티
# =============================================================================

def get_benchmark_config(name: str) -> dict[str, Any]:
    """벤치마크 설정 가져오기"""
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    """사용 가능한 벤치마크 목록"""
    return list(BENCHMARKS.keys())

