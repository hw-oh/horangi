"""KoHellaSwag - Korean HellaSwag Benchmark"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.hellaswag.hellaswag",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHellaSwag_mini:w5y3uB67dxszTK1uXakGqD2IYKZSrsW1AYQcPH9hIE8",
    split="train",
    solver="korean_multiple_choice",
    solver_args={
        "template": """다음 객관식 질문에 답하세요. 응답의 전체 내용은 '정답: $X'(쌍따옴표 제외) 형식이어야 합니다. 여기서 X는 {letters} 중 하나입니다.

{question}

{choices}"""
    },
    system_message=None
)
