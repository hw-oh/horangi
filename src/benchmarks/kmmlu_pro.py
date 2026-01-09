"""
KMMLU-Pro

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KMMLU_Pro_mini:Qbju8ttQj6C4HwI6N2UG7bqB1OnHTZ21IqluhZuiMsM",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    scorer="choice",
    system_message="이 문제는 전문 분야의 객관식 문제입니다. 추론 과정을 간결하게 요약한 후 답하세요.",
)
