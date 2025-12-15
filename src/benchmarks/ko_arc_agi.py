"""
Ko-ARC-AGI

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/ARC-AGI:GY9JPX8JljVpzuUsmS8p95KaVhiPMacxYdI59Z3HUXU",
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
