"""IFEval-Ko - 한국어 IFEval 벤치마크"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.ifeval.ifeval",
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/IFEval-Ko:SGtm8r2dBuXUnkS402O7vYwzrBGrzYCfrLu7WS2u2No",
)
