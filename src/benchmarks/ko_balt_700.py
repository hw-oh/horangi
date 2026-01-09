"""
KoBALT-700 - Korean MCQA Benchmark

Independent benchmark (no base) - solver/scorer directly specified
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoBALT_700_mini:RXgDQTYja0ZySmuQhH0xRmEA36UJPH7YQcf1LrpD9o0",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="letter",  # Data's answer is already a letter (A, B, C, ...)
    solver="korean_multiple_choice",
    scorer="choice",
    system_message="이 문제는 언어학 관련 문제입니다.",
)
