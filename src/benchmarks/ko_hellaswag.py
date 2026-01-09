"""KoHellaSwag - Korean HellaSwag Benchmark"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.hellaswag.hellaswag",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/KoHellaSwag_mini:w5y3uB67dxszTK1uXakGqD2IYKZSrsW1AYQcPH9hIE8",
    split="train",
    solver="korean_multiple_choice",
)
