"""
HAERAE_BENCH_V1-RC

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/HAERAE_BENCH_V1-RC:L33rg7qYEFD5kgvla9Atnj61MlFmFYHNmqwGvMZY5Yw",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="index_1",
    solver="multiple_choice",
    scorer="choice",
    system_message="주어진 질문에 가장 적절한 답을 선택하세요.",
)
