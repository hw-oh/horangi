"""IFEval-Ko - 한국어 IFEval 벤치마크"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.ifeval.ifeval",
    data_type="weave",
    data_source="weave:///horangi/horangi4/object/IFEval_Ko_mini:qzHRd8tmmARVui2M4dj4P363Ha8L28XQlvvcrUlrHCM",
)
