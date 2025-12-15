"""
KoTruthfulQA

독립 벤치마크 (base 없음) - solver/scorer 직접 지정
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/KoTruthfulQA-MC1:30sJ4YsJDrUS14DzBqZhXuZy7wiAai3LU9bpYpH6cJY",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",
    },
    answer_format="letter",
    solver="multiple_choice",
    scorer="choice",
    system_message="주어진 질문에 가장 적절한 답을 선택하세요.",
)
