"""
Ko-ARC-AGI

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoARC_AGI_mini:HSzsUWJnTXMYwOtS8A6wyfHM1DqsoTugtpBOwmvBuoA",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    answer_format="identity",
    solver="generate",
    scorer="grid_match",
    system_message="주어진 ARC 문제의 패턴을 분석하고, 출력 그리드를 정확히 예측하세요.",
)
