"""
KoTruthfulQA

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoTruthfulQA_mini:aXWwop2uqxplEhdvz576gyfUO4NSkrGNko7hguxueic",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="letter",
    solver="korean_multiple_choice",
    scorer="choice",
    system_message="이 문제는 진실성을 평가하는 문제입니다.",
)
