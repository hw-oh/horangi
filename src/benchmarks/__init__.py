"""
한국어 LLM 평가 벤치마크 정의

각 벤치마크는 개별 파일로 관리됩니다.
새 벤치마크를 추가하려면:
1. 이 폴더에 새 파일 생성 (예: ko_new.py)
2. CONFIG 딕셔너리 정의
3. 이 파일의 BENCHMARKS에 추가
4. horangi.py에 @task 함수 추가
"""

from benchmarks.ko_hellaswag import CONFIG as ko_hellaswag
from benchmarks.ko_aime2025 import CONFIG as ko_aime2025
from benchmarks.ifeval_ko import CONFIG as ifeval_ko
from benchmarks.ko_balt_700 import CONFIG as ko_balt_700
from benchmarks.haerae_bench_v1_rc import CONFIG as haerae_bench_v1_rc
from benchmarks.haerae_bench_v1_wo_rc import CONFIG as haerae_bench_v1_wo_rc
from benchmarks.kmmlu import CONFIG as kmmlu
from benchmarks.kmmlu_pro import CONFIG as kmmlu_pro
from benchmarks.squad_kor_v1 import CONFIG as squad_kor_v1
from benchmarks.ko_truthful_qa import CONFIG as ko_truthful_qa
from benchmarks.ko_moral import CONFIG as ko_moral
from benchmarks.ko_arc_agi import CONFIG as ko_arc_agi
from benchmarks.ko_gsm8k import CONFIG as ko_gsm8k
from benchmarks.korean_hate_speech import CONFIG as korean_hate_speech
from benchmarks.kobbq import CONFIG as kobbq
from benchmarks.ko_hle import CONFIG as ko_hle
# HalluLens 벤치마크
from benchmarks.ko_hallulens_wikiqa import CONFIG as ko_hallulens_wikiqa
from benchmarks.ko_hallulens_longwiki import CONFIG as ko_hallulens_longwiki
from benchmarks.ko_hallulens_generated import CONFIG as ko_hallulens_generated
from benchmarks.ko_hallulens_mixed import CONFIG as ko_hallulens_mixed
from benchmarks.ko_hallulens_nonexistent import CONFIG as ko_hallulens_nonexistent
# BFCL 벤치마크
from benchmarks.bfcl import CONFIG as bfcl
from benchmarks.bfcl_extended import CONFIG as bfcl_extended  # 하위 호환성
from benchmarks.bfcl_text import CONFIG as bfcl_text  # 하위 호환성
# MT-Bench
from benchmarks.mtbench_ko import CONFIG as mtbench_ko
# SWE-bench
from benchmarks.swebench_verified_official_80 import CONFIG as swebench_verified_official_80

# 벤치마크 설명
BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    # 일반
    "ko_hellaswag": "상식 추론 (문장 완성)",
    "ko_aime2025": "AIME 2025 수학 문제",
    "ifeval_ko": "지시 따르기 평가",
    "ko_balt_700": "언어 이해 및 추론",
    # 지식
    "haerae_bench_v1_rc": "HAERAE v1 (독해 포함)",
    "haerae_bench_v1_wo_rc": "HAERAE v1 (독해 제외)",
    "kmmlu": "한국어 MMLU",
    "kmmlu_pro": "한국어 MMLU Pro (고난도)",
    "squad_kor_v1": "한국어 독해 QA",
    "ko_truthful_qa": "사실성 평가",
    # 추론
    "ko_moral": "도덕적 판단",
    "ko_arc_agi": "ARC-AGI 추론 (그리드)",
    "ko_gsm8k": "초등 수학 문제",
    # 편향/안전
    "korean_hate_speech": "혐오 발언 탐지",
    "kobbq": "편향성 판단 (BBQ)",
    "ko_hle": "Humanity's Last Exam",
    # HalluLens (환각)
    "ko_hallulens_wikiqa": "위키 QA 환각 평가",
    "ko_hallulens_longwiki": "긴 문서 QA 환각 평가",
    "ko_hallulens_generated": "가상 엔티티 거부 (생성)",
    "ko_hallulens_mixed": "가상 엔티티 거부 (혼합)",
    "ko_hallulens_nonexistent": "가상 엔티티 거부 (통합)",
    # Function Calling
    "bfcl": "함수 호출 (통합, 모델 설정에 따라 자동 선택)",
    "bfcl_extended": "함수 호출 (Native Tool) [deprecated]",
    "bfcl_text": "함수 호출 (Text-based) [deprecated]",
    # 대화
    "mtbench_ko": "멀티턴 대화 평가",
    # 코딩
    "swebench_verified_official_80": "SWE-bench 버그 수정 (80개)",
}

# 모든 벤치마크 설정
BENCHMARKS: dict = {
    "ko_hellaswag": ko_hellaswag,
    "ko_aime2025": ko_aime2025,
    "ifeval_ko": ifeval_ko,
    "ko_balt_700": ko_balt_700,
    "haerae_bench_v1_rc": haerae_bench_v1_rc,
    "haerae_bench_v1_wo_rc": haerae_bench_v1_wo_rc,
    "kmmlu": kmmlu,
    "kmmlu_pro": kmmlu_pro,
    "squad_kor_v1": squad_kor_v1,
    "ko_truthful_qa": ko_truthful_qa,
    "ko_moral": ko_moral,
    "ko_arc_agi": ko_arc_agi,
    "ko_gsm8k": ko_gsm8k,
    "korean_hate_speech": korean_hate_speech,
    "kobbq": kobbq,
    "ko_hle": ko_hle,
    # HalluLens 벤치마크
    "ko_hallulens_wikiqa": ko_hallulens_wikiqa,
    "ko_hallulens_longwiki": ko_hallulens_longwiki,
    "ko_hallulens_generated": ko_hallulens_generated,
    "ko_hallulens_mixed": ko_hallulens_mixed,
    "ko_hallulens_nonexistent": ko_hallulens_nonexistent,
    # BFCL 벤치마크
    "bfcl": bfcl,
    "bfcl_extended": bfcl_extended,  # 하위 호환성
    "bfcl_text": bfcl_text,  # 하위 호환성
    # MT-Bench
    "mtbench_ko": mtbench_ko,
    # SWE-bench
    "swebench_verified_official_80": swebench_verified_official_80,
}

def get_benchmark_config(name: str) -> dict:
    """벤치마크 설정 가져오기 (dict 반환)"""
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    
    config = BENCHMARKS[name]
    
    # BenchmarkConfig dataclass인 경우 dict로 변환
    if hasattr(config, "to_dict"):
        return config.to_dict()
    
    # 이미 dict인 경우 그대로 반환
    return config


def list_benchmarks() -> list[str]:
    """사용 가능한 벤치마크 목록"""
    return list(BENCHMARKS.keys())


def list_benchmarks_with_descriptions() -> list[tuple[str, str]]:
    """벤치마크 목록과 설명 반환"""
    return [
        (name, BENCHMARK_DESCRIPTIONS.get(name, ""))
        for name in BENCHMARKS.keys()
    ]


__all__ = [
    "BENCHMARKS",
    "BENCHMARK_DESCRIPTIONS",
    "get_benchmark_config",
    "list_benchmarks",
    "list_benchmarks_with_descriptions",
]
