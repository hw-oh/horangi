"""KoHellaSwag - 한국어 HellaSwag 벤치마크"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    base="inspect_evals.hellaswag.hellaswag",
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/KoHellaSwag:PY229AMRxLFoCLqKsaEguY4jVCuvyoMnQ5wJ1wkrXfU",
    split="train",
)
