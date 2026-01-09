"""
HAERAE_BENCH_V1-RC

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
의미해석(Reading Comprehension) 능력 평가
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HAERAE_Bench_v1_RC_mini:VOB5YJUpzgAT1XqJZXj4ps9jfXssBq853dLLKqp8Ops",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    scorer="choice",
)
