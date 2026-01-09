"""
HAERAE_BENCH_V1

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:AUDj1Yc8irM87b4DOXS9LK31AXfCPo8Uh8aEXyGa9J4",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="korean_multiple_choice",
    scorer="choice",
    sampling="balanced",
    sampling_by="category",
)
