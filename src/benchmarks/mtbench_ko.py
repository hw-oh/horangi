"""
KoMT-Bench - 한국어 멀티턴 대화 평가 벤치마크

원본: https://huggingface.co/datasets/LGAI-EXAONE/KoMT-Bench (LG AI Research)

8개 카테고리 (writing, roleplay, reasoning, math, coding, extraction, stem, humanities)
각 카테고리당 10개 질문, 총 80개
2턴 대화 형식으로 LLM Judge가 1-10점 평가
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/KoMT_Bench:latest",
    field_mapping={
        "id": "id",
        "input": "turn1",  # 첫 번째 질문이 input
        # target은 없음 - LLM Judge가 평가
    },
    answer_format="identity",
    solver="mtbench_solver",
    scorer="mtbench_scorer",
    # system_message는 solver에서 처리하지 않음 (질문만 전달)
)
