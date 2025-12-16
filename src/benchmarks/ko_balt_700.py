"""
KoBALT-700 - 한국어 MCQA 벤치마크

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
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
    answer_format="letter",  # 데이터의 answer가 이미 문자(A, B, C, ...)임
    solver="multiple_choice",
    scorer="choice",
    system_message="주어진 질문에 가장 적절한 답을 선택하세요.",
)
